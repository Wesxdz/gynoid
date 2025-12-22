#include "blender_panel.h"
#include "debug_log.h"
#include <sstream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// Simple JSON parser for Blender panel data
BlenderPanelData parse_blender_panel_json(const std::string& json_str) {
    BlenderPanelData data;

    LOG_TRACE(LogCategory::BLENDER_PANEL, "Parsing JSON of length {}", json_str.length());
    LOG_TRACE(LogCategory::BLENDER_PANEL, "First 200 chars: {}", json_str.substr(0, std::min((size_t)200, json_str.length())));

    auto extract_string = [](const std::string& json, const std::string& key) -> std::string {
        std::string search = "\"" + key + "\": \"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos += search.length();
        size_t end = json.find("\"", pos);
        return json.substr(pos, end - pos);
    };

    auto extract_int = [](const std::string& json, const std::string& key) -> int {
        std::string search = "\"" + key + "\": ";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return 0;
        pos += search.length();
        size_t end = json.find_first_of(",}", pos);
        return std::stoi(json.substr(pos, end - pos));
    };

    // Extract basic info
    data.container_id = extract_string(json_str, "container_id");
    data.timestamp = extract_string(json_str, "timestamp");
    data.quadrant = blender_container_id_to_quadrant(data.container_id);

    LOG_DEBUG(LogCategory::BLENDER_PANEL, "container_id={}, quadrant={}", data.container_id, data.quadrant);

    // Extract panels array
    size_t panels_start = json_str.find("\"panels\": [");
    if (panels_start != std::string::npos) {
        LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");

        // Find the matching closing bracket for the panels array
        size_t bracket_pos = json_str.find("[", panels_start);
        int bracket_count = 1;
        size_t scan_pos = bracket_pos + 1;
        size_t panels_end = std::string::npos;

        while (scan_pos < json_str.length() && bracket_count > 0) {
            if (json_str[scan_pos] == '[') bracket_count++;
            else if (json_str[scan_pos] == ']') {
                bracket_count--;
                if (bracket_count == 0) {
                    panels_end = scan_pos;
                    break;
                }
            }
            scan_pos++;
        }

        if (panels_end == std::string::npos) {
            LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");
            return data;
        }

        std::string panels_section = json_str.substr(panels_start, panels_end - panels_start);
        LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");

        // Find each panel object
        size_t pos = 0;
        int panel_count = 0;
        while ((pos = panels_section.find("{", pos)) != std::string::npos) {
            LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");

            // Find matching closing brace for this panel object
            int brace_count = 1;
            size_t scan_pos = pos + 1;
            size_t end = std::string::npos;

            while (scan_pos < panels_section.length() && brace_count > 0) {
                if (panels_section[scan_pos] == '{') brace_count++;
                else if (panels_section[scan_pos] == '}') {
                    brace_count--;
                    if (brace_count == 0) {
                        end = scan_pos;
                        break;
                    }
                }
                scan_pos++;
            }

            if (end == std::string::npos) {
                LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");
                break;
            }

            std::string panel_str = panels_section.substr(pos, end - pos + 1);
            LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");

            BlenderPanelBounds panel;
            panel.type = extract_string(panel_str, "type");
            panel.x = extract_int(panel_str, "x");
            panel.y = extract_int(panel_str, "y");
            panel.width = extract_int(panel_str, "width");
            panel.height = extract_int(panel_str, "height");

            // Parse regions array
            size_t regions_start = panel_str.find("\"regions\": [");
            if (regions_start != std::string::npos) {
                size_t regions_end = panel_str.find("]", regions_start);
                std::string regions_section = panel_str.substr(regions_start, regions_end - regions_start);

                // Find each region object
                size_t region_pos = 0;
                while ((region_pos = regions_section.find("{", region_pos)) != std::string::npos) {
                    size_t region_end = regions_section.find("}", region_pos);
                    if (region_end == std::string::npos) break;

                    std::string region_str = regions_section.substr(region_pos, region_end - region_pos + 1);

                    BlenderRegionBounds region;
                    region.type = extract_string(region_str, "type");
                    region.x = extract_int(region_str, "x");
                    region.y = extract_int(region_str, "y");
                    region.width = extract_int(region_str, "width");
                    region.height = extract_int(region_str, "height");

                    panel.regions.push_back(region);
                    region_pos = region_end + 1;
                }
            }

            data.panels.push_back(panel);
            panel_count++;
            LOG_TRACE(LogCategory::BLENDER_PANEL, "Added panel {}: type={}, regions={}", panel_count, panel.type, panel.regions.size());
            pos = end + 1;
        }
        LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");
    } else {
        LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");
    }

    LOG_TRACE(LogCategory::BLENDER_PANEL, "Parser trace");
    return data;
}

void blender_socket_server_thread(flecs::world* ecs, int port, std::atomic<bool>* running) {
    LOG_DEBUG(LogCategory::BLENDER_PANEL, "Socket thread starting...");

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        LOG_ERROR(LogCategory::BLENDER_PANEL, "Failed to create socket: {}", strerror(errno));
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        LOG_ERROR(LogCategory::BLENDER_PANEL, "Failed to bind to port {}: {}", port, strerror(errno));
        close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        LOG_ERROR(LogCategory::BLENDER_PANEL, "Failed to listen: {}", strerror(errno));
        close(server_fd);
        return;
    }

    LOG_INFO(LogCategory::BLENDER_PANEL, "Socket server listening on port {}", port);

    // Get quadrant panel entities
    auto q0 = ecs->lookup("BlenderQuadrant0");
    auto q1 = ecs->lookup("BlenderQuadrant1");
    auto q2 = ecs->lookup("BlenderQuadrant2");
    auto q3 = ecs->lookup("BlenderQuadrant3");

    while (*running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            continue;
        }

        // Receive JSON data
        char buffer[65536];
        int bytes_received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';
            std::string json_str(buffer);

            // Parse JSON
            BlenderPanelData panel_data = parse_blender_panel_json(json_str);

            int total_regions = 0;
            for (const auto& panel : panel_data.panels) {
                total_regions += panel.regions.size();
            }

            LOG_DEBUG(LogCategory::BLENDER_PANEL, "Received data from {} (quadrant {}) with {} panels and {} total regions",
                     panel_data.container_id, panel_data.quadrant, panel_data.panels.size(), total_regions);

            // Update appropriate quadrant
            if (panel_data.quadrant == 0 && q0.is_alive()) {
                q0.ensure<BlenderQuadrantPanels>().update(panel_data);
            } else if (panel_data.quadrant == 1 && q1.is_alive()) {
                q1.ensure<BlenderQuadrantPanels>().update(panel_data);
            } else if (panel_data.quadrant == 2 && q2.is_alive()) {
                q2.ensure<BlenderQuadrantPanels>().update(panel_data);
            } else if (panel_data.quadrant == 3 && q3.is_alive()) {
                q3.ensure<BlenderQuadrantPanels>().update(panel_data);
            }
        }

        close(client_fd);
    }

    close(server_fd);
}

void BlenderPanelModule(flecs::world& ecs) {
    LOG_DEBUG(LogCategory::BLENDER_PANEL, "Initializing module...");

    // Register components
    ecs.component<BlenderPanelServer>();
    ecs.component<BlenderQuadrantPanels>();
    LOG_DEBUG(LogCategory::BLENDER_PANEL, "Components registered");

    // Create quadrant panel entities
    ecs.entity("BlenderQuadrant0").ensure<BlenderQuadrantPanels>();
    ecs.entity("BlenderQuadrant1").ensure<BlenderQuadrantPanels>();
    ecs.entity("BlenderQuadrant2").ensure<BlenderQuadrantPanels>();
    ecs.entity("BlenderQuadrant3").ensure<BlenderQuadrantPanels>();
    LOG_DEBUG(LogCategory::BLENDER_PANEL, "Quadrant entities created");

    // Create socket server entity
    auto server_entity = ecs.entity("BlenderPanelSocketServer");
    auto& server = server_entity.ensure<BlenderPanelServer>();
    server.port = 9998;
    server.running = true;
    LOG_DEBUG(LogCategory::BLENDER_PANEL, "Server entity created, starting thread...");

    // Start socket server thread
    std::atomic<bool>* running_flag = new std::atomic<bool>(true);
    server.server_thread = std::thread(blender_socket_server_thread, &ecs, server.port, running_flag);

    LOG_INFO(LogCategory::BLENDER_PANEL, "Module initialized on port {}", server.port);
}
