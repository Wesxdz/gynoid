#!/bin/bash
# setup_wifi_passthrough.sh
REMOTE_HOST="${1:-192.168.1.104}" # Default to your server if not provided

# 1. Dynamically find the Phone interface
TETHER_IFACE=$(nmcli -t -f DEVICE,STATE,TYPE device | grep ":connected:ethernet" | grep "enx" | cut -d: -f1 | head -n 1)

if [ -z "$TETHER_IFACE" ]; then
    echo "Error: No phone tether found. Ensure USB tethering is ON."
    exit 1
fi

echo "Detected Phone Interface: $TETHER_IFACE"

# 2. Enable Forwarding & NAT
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -F POSTROUTING
sudo iptables -t nat -A POSTROUTING -o "$TETHER_IFACE" -j MASQUERADE

# 3. Fix the Remote Gateway
echo "Updating gateway on remote server $REMOTE_HOST..."
sshpass -p 'GrokValentine42!' ssh -o StrictHostKeyChecking=no grok@$REMOTE_HOST \
    "sudo ip route del default || true; sudo ip route add default via 192.168.1.1"

echo "Passthrough Restored via $TETHER_IFACE!"
