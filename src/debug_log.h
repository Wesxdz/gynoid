#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <flecs.h>

// Log categories - each gets its own logger instance
enum class LogCategory {
    OCR_STREAM,
    OCR_RENDER,
    SERVER_LAUNCHER,
    X11_OUTLINE,
    BLENDER_PANEL,
    VNC_CLIENT,
    VISION_PROCESSOR,
    RENDER_QUEUE,
    INPUT,
    SYSTEM,
    NETWORK
};

// Category-based logger manager using spdlog
class DebugLogger {
private:
    std::unordered_map<LogCategory, std::shared_ptr<spdlog::logger>> loggers;
    std::unordered_map<LogCategory, std::string> category_names;
    mutable std::mutex mutex;

    DebugLogger() {
        // Initialize category names
        category_names[LogCategory::OCR_STREAM] = "OCR_STREAM";
        category_names[LogCategory::OCR_RENDER] = "OCR_RENDER";
        category_names[LogCategory::SERVER_LAUNCHER] = "SERVER_LAUNCHER";
        category_names[LogCategory::X11_OUTLINE] = "X11_OUTLINE";
        category_names[LogCategory::BLENDER_PANEL] = "BLENDER_PANEL";
        category_names[LogCategory::VNC_CLIENT] = "VNC_CLIENT";
        category_names[LogCategory::VISION_PROCESSOR] = "VISION_PROCESSOR";
        category_names[LogCategory::RENDER_QUEUE] = "RENDER_QUEUE";
        category_names[LogCategory::INPUT] = "INPUT";
        category_names[LogCategory::SYSTEM] = "SYSTEM";
        category_names[LogCategory::NETWORK] = "NETWORK";

        // Create console sink with colors
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%H:%M:%S.%e] [%n] [%^%l%$] %v");

        // Create loggers for each category
        for (const auto& pair : category_names) {
            auto logger = std::make_shared<spdlog::logger>(pair.second, console_sink);
            loggers[pair.first] = logger;

            // Default: only enable important categories
            if (pair.first == LogCategory::X11_OUTLINE ||
                pair.first == LogCategory::BLENDER_PANEL) {
                logger->set_level(spdlog::level::debug);  // Show DEBUG logs for OCR
            } else if (pair.first == LogCategory::X11_OUTLINE ||
                       pair.first == LogCategory::SYSTEM) {
                logger->set_level(spdlog::level::info);
            } else {
                logger->set_level(spdlog::level::off);
            }
        }
    }

public:
    static DebugLogger& instance() {
        static DebugLogger logger;
        return logger;
    }

    std::shared_ptr<spdlog::logger> get(LogCategory category) {
        std::lock_guard<std::mutex> lock(mutex);
        return loggers[category];
    }

    void enable_category(LogCategory category, spdlog::level::level_enum level = spdlog::level::debug) {
        std::lock_guard<std::mutex> lock(mutex);
        if (loggers.find(category) != loggers.end()) {
            loggers[category]->set_level(level);
        }
    }

    void disable_category(LogCategory category) {
        std::lock_guard<std::mutex> lock(mutex);
        if (loggers.find(category) != loggers.end()) {
            loggers[category]->set_level(spdlog::level::off);
        }
    }

    void enable_all(spdlog::level::level_enum level = spdlog::level::debug) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& pair : loggers) {
            pair.second->set_level(level);
        }
    }

    void disable_all() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& pair : loggers) {
            pair.second->set_level(spdlog::level::off);
        }
    }

    void set_level(LogCategory category, spdlog::level::level_enum level) {
        std::lock_guard<std::mutex> lock(mutex);
        if (loggers.find(category) != loggers.end()) {
            loggers[category]->set_level(level);
        }
    }

    void print_status() {
        std::lock_guard<std::mutex> lock(mutex);
        auto system_logger = loggers[LogCategory::SYSTEM];
        system_logger->info("=== Debug Logger Status ===");
        for (const auto& pair : category_names) {
            auto logger = loggers[pair.first];
            bool enabled = logger->level() != spdlog::level::off;
            std::string status = enabled ? "ON " : "OFF";
            std::string level_str = spdlog::level::to_string_view(logger->level()).data();
            system_logger->info("  {:<20} {} ({})", pair.second, status, level_str);
        }
        system_logger->info("===========================");
    }
};

// Convenience macros that use spdlog
#define LOG_TRACE(category, ...) DebugLogger::instance().get(category)->trace(__VA_ARGS__)
#define LOG_DEBUG(category, ...) DebugLogger::instance().get(category)->debug(__VA_ARGS__)
#define LOG_INFO(category, ...) DebugLogger::instance().get(category)->info(__VA_ARGS__)
#define LOG_WARN(category, ...) DebugLogger::instance().get(category)->warn(__VA_ARGS__)
#define LOG_ERROR(category, ...) DebugLogger::instance().get(category)->error(__VA_ARGS__)
#define LOG_CRITICAL(category, ...) DebugLogger::instance().get(category)->critical(__VA_ARGS__)

// Flecs component for logger control
struct DebugLogConfig {
    bool show_ui = false;  // For future ImGui integration
};

// Initialize logger module
inline void DebugLogModule(flecs::world& ecs) {
    ecs.component<DebugLogConfig>();

    auto config_entity = ecs.entity("DebugLogConfig")
        .add<DebugLogConfig>();

    LOG_INFO(LogCategory::SYSTEM, "Debug logger initialized (using spdlog)");
    DebugLogger::instance().print_status();
}
