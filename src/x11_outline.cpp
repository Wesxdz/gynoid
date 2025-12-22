#include "x11_outline.h"
#include "debug_log.h"
#include <sstream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "vnc_struct.h"

// Simple JSON parser for our specific format
X11OutlineData parse_x11_json(const std::string& json_str) {
    X11OutlineData data;

    // Very simple JSON parsing - just extract the fields we need
    // This is a simplified parser; for production you'd want a proper JSON library

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

    auto extract_float = [](const std::string& json, const std::string& key) -> float {
        std::string search = "\"" + key + "\": ";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return 0.0f;
        pos += search.length();
        size_t end = json.find_first_of(",}", pos);
        return std::stof(json.substr(pos, end - pos));
    };

    // Extract basic info
    data.container_id = extract_string(json_str, "container_id");
    data.timestamp = extract_string(json_str, "timestamp");

    // Extract screen dimensions
    data.screen.width = extract_int(json_str, "width");
    data.screen.height = extract_int(json_str, "height");
    data.screen.x = 0;
    data.screen.y = 0;

    // Extract work area if present
    if (json_str.find("\"work_area\"") != std::string::npos) {
        data.has_work_area = true;
        // Find work_area section
        size_t wa_start = json_str.find("\"work_area\"");
        size_t wa_end = json_str.find("}", wa_start);
        std::string wa_section = json_str.substr(wa_start, wa_end - wa_start);
        data.work_area.x = extract_int(wa_section, "x");
        data.work_area.y = extract_int(wa_section, "y");
        data.work_area.width = extract_int(wa_section, "width");
        data.work_area.height = extract_int(wa_section, "height");
    }

    // Extract visible windows
    size_t vw_start = json_str.find("\"visible_windows\": [");
    if (vw_start != std::string::npos) {
        size_t vw_end = json_str.find("]", vw_start);
        std::string vw_section = json_str.substr(vw_start, vw_end - vw_start);

        // Find each window object
        size_t pos = 0;
        while ((pos = vw_section.find("{", pos)) != std::string::npos) {
            size_t end = vw_section.find("}", pos);
            if (end == std::string::npos) break;

            std::string win_str = vw_section.substr(pos, end - pos + 1);

            X11WindowData win;
            win.id = extract_string(win_str, "id");
            win.name = extract_string(win_str, "name");
            win.z_order = extract_int(win_str, "z_order");

            // Extract clipped bounds (preferred) or regular bounds as fallback
            size_t cb_start = win_str.find("\"clipped_bounds\"");
            if (cb_start != std::string::npos) {
                size_t cb_end = win_str.find("}", cb_start);
                std::string cb_str = win_str.substr(cb_start, cb_end - cb_start);
                win.bounds.x = extract_int(cb_str, "x");
                win.bounds.y = extract_int(cb_str, "y");
                win.bounds.width = extract_int(cb_str, "width");
                win.bounds.height = extract_int(cb_str, "height");
            } else {
                // Fallback to regular bounds if no clipped bounds
                size_t b_start = win_str.find("\"bounds\"");
                if (b_start != std::string::npos) {
                    size_t b_end = win_str.find("}", b_start);
                    std::string bounds_str = win_str.substr(b_start, b_end - b_start);
                    win.bounds.x = extract_int(bounds_str, "x");
                    win.bounds.y = extract_int(bounds_str, "y");
                    win.bounds.width = extract_int(bounds_str, "width");
                    win.bounds.height = extract_int(bounds_str, "height");
                }
            }

            // Extract visibility info
            size_t v_start = win_str.find("\"visibility\"");
            if (v_start != std::string::npos) {
                size_t v_end = win_str.find("}", v_start);
                std::string vis_str = win_str.substr(v_start, v_end - v_start);
                win.visible_pixels = extract_int(vis_str, "visible_pixels");
                win.onscreen_pixels = extract_int(vis_str, "onscreen_pixels");
                win.visibility_percent = extract_float(vis_str, "percent");
            }

            data.visible_windows.push_back(win);
            pos = end + 1;
        }
    }

    return data;
}

void socket_server_thread(flecs::world* ecs, int port, std::atomic<bool>* running, X11OutlineQueue* queue) {
    LOG_DEBUG(LogCategory::X11_OUTLINE, "Socket thread starting...");

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        LOG_ERROR(LogCategory::X11_OUTLINE, "Failed to create socket: {}", strerror(errno));
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        LOG_ERROR(LogCategory::X11_OUTLINE, "Failed to bind to port {}: {}", port, strerror(errno));
        close(server_fd);
        return;
    }

    if (listen(server_fd, 10) < 0) {
        LOG_ERROR(LogCategory::X11_OUTLINE, "Failed to listen: {}", strerror(errno));
        close(server_fd);
        return;
    }

    LOG_INFO(LogCategory::X11_OUTLINE, "Socket server listening on port {}", port);

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
            X11OutlineData outline_data = parse_x11_json(json_str);

            LOG_DEBUG(LogCategory::X11_OUTLINE, "Received data from {} with {} windows",
                     outline_data.container_id, outline_data.visible_windows.size());

            // Push to thread-safe queue for processing by Flecs system
            queue->push(outline_data);
        }

        close(client_fd);
    }

    close(server_fd);
}

void X11OutlineModule(flecs::world& ecs) {
    LOG_DEBUG(LogCategory::X11_OUTLINE, "Initializing module...");

    // Register components
    ecs.component<X11OutlineServer>();
    ecs.component<X11Container>();
    ecs.component<X11WindowInfo>()
        .member<std::string>("id")
        .member<std::string>("name");
    ecs.component<X11WindowBounds>()
        .member<int>("x")
        .member<int>("y")
        .member<int>("width")
        .member<int>("height");
    ecs.component<X11WindowVisibility>();
    ecs.component<X11VisibleWindow>();
    ecs.component<X11SpecialWindow>();
    LOG_DEBUG(LogCategory::X11_OUTLINE, "Components registered");

    // Create queue as a static variable (shared between thread and system)
    static X11OutlineQueue global_queue;

    // Create socket server entity
    auto server_entity = ecs.entity("X11OutlineSocketServer");
    auto& server = server_entity.ensure<X11OutlineServer>();
    server.port = 5299;
    server.running = true;
    server.queue = &global_queue;
    LOG_DEBUG(LogCategory::X11_OUTLINE, "Server entity created, starting thread...");

    // Start socket server thread
    std::atomic<bool>* running_flag = new std::atomic<bool>(true);
    server.server_thread = std::thread(socket_server_thread, &ecs, server.port, running_flag, server.queue);

    // Add system to process queued X11 outline data
    ecs.system<X11Container>()
        .kind(flecs::PreUpdate)
        .each([&](flecs::entity e, X11Container& container_comp) {
            auto pending_updates = global_queue.pop_all();
            
            for (const auto& outline_data : pending_updates) {
                if (outline_data.container_id != "1") continue;
                std::cout << "Container id is " << outline_data.container_id << std::endl;

                container_comp.container_id = outline_data.container_id;
                container_comp.screen = outline_data.screen;
                container_comp.work_area = outline_data.work_area;
                container_comp.has_work_area = outline_data.has_work_area;
                container_comp.last_update_timestamp = outline_data.timestamp;

                // Clear old window entities for this container
                auto old_windows = ecs.query_builder()
                    .with(flecs::ChildOf, e)
                    .build();
                old_windows.each([](flecs::entity e) {
                    e.destruct();
                });

                // Create window entities for visible windows
                for (const auto& win_data : outline_data.visible_windows) {
                    auto win_entity = ecs.entity()
                        .child_of(e)
                        .add<X11VisibleWindow>();

                    // Set window info
                    auto& info = win_entity.ensure<X11WindowInfo>();
                    info.id = win_data.id;
                    info.name = win_data.name;
                    info.z_order = win_data.z_order;

                    // Set window bounds
                    win_entity.set<X11WindowBounds>(win_data.bounds);

                    // Set visibility metrics
                    win_entity.set<X11WindowVisibility>({
                        win_data.visible_pixels,
                        win_data.onscreen_pixels,
                        win_data.visibility_percent
                    });
                }

                // Create window entities for special windows
                for (const auto& win_data : outline_data.special_windows) {
                    auto win_entity = ecs.entity()
                        .child_of(e)
                        .add<X11SpecialWindow>();

                    auto& info = win_entity.ensure<X11WindowInfo>();
                    info.id = win_data.id;
                    info.name = win_data.name;
                    info.z_order = win_data.z_order;

                    win_entity.set<X11WindowBounds>(win_data.bounds);

                    win_entity.set<X11WindowVisibility>({
                        win_data.visible_pixels,
                        win_data.onscreen_pixels,
                        win_data.visibility_percent
                    });
                }
            }
        });

    LOG_INFO(LogCategory::X11_OUTLINE, "Module initialized on port {}", server.port);
}
