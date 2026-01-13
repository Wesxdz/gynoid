#!/bin/bash
# setup_wifi_passthrough.sh
REMOTE_HOST="${1:-192.168.1.104}" # Default to your server if not provided
LOCAL_IFACE="${2:-enp39s0}"       # Interface connected to remote network
DNS_SERVER="${3:-8.8.8.8}"        # DNS server for remote to use
REMOTE_PASS='GrokValentine42!'    # Password for remote sudo

# 1. Dynamically find the Phone tether interface
# USB tethering creates enx* interfaces, but we need to exclude altnames of other interfaces
# Look for enx* interfaces that have their own IP in a different subnet than LOCAL_IFACE
TETHER_IFACE=""
for iface in $(ip -o link show | grep -oP '\d+: \Kenx[a-f0-9]+(?=:)'); do
    # Check if this interface has an IP address (not just an altname)
    if ip -4 addr show dev "$iface" 2>/dev/null | grep -q 'inet '; then
        # Make sure it's not on the local network (192.168.1.x)
        IFACE_IP=$(ip -4 addr show dev "$iface" | grep -oP 'inet \K[0-9.]+')
        if [[ ! "$IFACE_IP" =~ ^192\.168\.1\. ]]; then
            TETHER_IFACE="$iface"
            break
        fi
    fi
done

# Fallback: check nmcli for any connected enx interface not matching local subnet
if [ -z "$TETHER_IFACE" ]; then
    for iface in $(nmcli -t -f DEVICE,STATE device | grep ":connected" | cut -d: -f1); do
        if [[ "$iface" =~ ^enx ]] && ip -4 addr show dev "$iface" 2>/dev/null | grep -q 'inet '; then
            IFACE_IP=$(ip -4 addr show dev "$iface" | grep -oP 'inet \K[0-9.]+')
            if [[ ! "$IFACE_IP" =~ ^192\.168\.1\. ]]; then
                TETHER_IFACE="$iface"
                break
            fi
        fi
    done
fi

if [ -z "$TETHER_IFACE" ]; then
    echo "Error: No phone tether found. Ensure USB tethering is ON."
    echo "Available interfaces:"
    ip -o link show | awk -F': ' '{print $2}'
    exit 1
fi

# Verify the interface is UP and has an IP
TETHER_IP=$(ip -4 addr show "$TETHER_IFACE" 2>/dev/null | grep -oP 'inet \K[0-9.]+')
if [ -z "$TETHER_IP" ]; then
    echo "Error: Tether interface $TETHER_IFACE has no IP address."
    exit 1
fi

echo "Detected Phone Interface: $TETHER_IFACE (IP: $TETHER_IP)"

# 2. Enable Forwarding & NAT
echo "Enabling IP forwarding..."
sudo sysctl -w net.ipv4.ip_forward=1

echo "Configuring NAT..."
sudo iptables -t nat -F POSTROUTING
sudo iptables -t nat -A POSTROUTING -o "$TETHER_IFACE" -j MASQUERADE

# 3. Allow forwarding between interfaces (critical if FORWARD policy is DROP)
echo "Configuring FORWARD chain..."
sudo iptables -A FORWARD -i "$LOCAL_IFACE" -o "$TETHER_IFACE" -j ACCEPT
sudo iptables -A FORWARD -i "$TETHER_IFACE" -o "$LOCAL_IFACE" -m state --state RELATED,ESTABLISHED -j ACCEPT

# 4. Fix the Remote Gateway and DNS
echo "Updating gateway and DNS on remote server $REMOTE_HOST..."
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no grok@$REMOTE_HOST bash -c "'
    echo \"$REMOTE_PASS\" | sudo -S ip route del default 2>/dev/null || true
    echo \"$REMOTE_PASS\" | sudo -S ip route add default via 192.168.1.1
    echo \"nameserver $DNS_SERVER\" | sudo -S tee /etc/resolv.conf > /dev/null
    echo \"Gateway and DNS configured.\"
'"

# 5. Verify connectivity from remote
echo "Testing connectivity from remote..."
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no grok@$REMOTE_HOST \
    "ping -c 2 -W 2 8.8.8.8 && echo 'Internet OK' || echo 'Internet FAILED'"

echo "Passthrough configured via $TETHER_IFACE!"
