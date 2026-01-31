/**
 * @file logging.hpp
 * @brief Caliby Logging System
 * 
 * Provides configurable logging with different log levels.
 * Thread-safe, lightweight logging for the Caliby vector database.
 */

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <atomic>

namespace caliby {

/**
 * Log levels for the Caliby logging system.
 * Ordered from most verbose (DEBUG) to least verbose (OFF).
 */
enum class LogLevel {
    DEBUG = 0,   // Detailed debugging information
    INFO = 1,    // General informational messages
    WARN = 2,    // Warning messages (recoverable issues)
    ERROR = 3,   // Error messages (serious issues)
    OFF = 4      // Disable all logging
};

/**
 * Convert LogLevel to string representation.
 */
inline const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::OFF:   return "OFF";
        default: return "UNKNOWN";
    }
}

/**
 * Parse string to LogLevel.
 */
inline LogLevel string_to_log_level(const std::string& str) {
    if (str == "DEBUG" || str == "debug") return LogLevel::DEBUG;
    if (str == "INFO" || str == "info") return LogLevel::INFO;
    if (str == "WARN" || str == "warn" || str == "WARNING" || str == "warning") return LogLevel::WARN;
    if (str == "ERROR" || str == "error") return LogLevel::ERROR;
    if (str == "OFF" || str == "off") return LogLevel::OFF;
    return LogLevel::INFO;  // Default to INFO
}

/**
 * Logger singleton class.
 * Thread-safe logging with configurable log level.
 */
class Logger {
public:
    /**
     * Get the singleton Logger instance.
     */
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    /**
     * Set the minimum log level.
     * Messages below this level will not be logged.
     */
    void set_level(LogLevel level) {
        current_level_.store(level, std::memory_order_relaxed);
    }

    /**
     * Get the current log level.
     */
    LogLevel get_level() const {
        return current_level_.load(std::memory_order_relaxed);
    }

    /**
     * Check if a given log level is enabled.
     */
    bool is_enabled(LogLevel level) const {
        return level >= current_level_.load(std::memory_order_relaxed);
    }

    /**
     * Enable or disable timestamps in log messages.
     */
    void set_show_timestamp(bool show) {
        show_timestamp_.store(show, std::memory_order_relaxed);
    }

    /**
     * Enable or disable the component/tag in log messages.
     */
    void set_show_component(bool show) {
        show_component_.store(show, std::memory_order_relaxed);
    }

    /**
     * Log a message at the specified level.
     */
    template<typename... Args>
    void log(LogLevel level, const char* component, Args&&... args) {
        if (!is_enabled(level)) {
            return;
        }

        std::ostringstream oss;
        
        // Optionally add timestamp
        if (show_timestamp_.load(std::memory_order_relaxed)) {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
            oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
                << '.' << std::setfill('0') << std::setw(3) << ms.count() << ' ';
        }

        // Add log level
        oss << '[' << log_level_to_string(level) << ']';
        
        // Optionally add component
        if (show_component_.load(std::memory_order_relaxed) && component && component[0] != '\0') {
            oss << '[' << component << ']';
        }

        oss << ' ';

        // Append all arguments
        ((oss << args), ...);

        // Thread-safe output
        std::lock_guard<std::mutex> lock(mutex_);
        if (level >= LogLevel::WARN) {
            std::cerr << oss.str() << std::endl;
        } else {
            std::cerr << oss.str() << std::endl;
        }
    }

private:
    Logger() 
        : current_level_(LogLevel::INFO)
        , show_timestamp_(false)
        , show_component_(true) {}
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::atomic<LogLevel> current_level_;
    std::atomic<bool> show_timestamp_;
    std::atomic<bool> show_component_;
    std::mutex mutex_;
};

// ============================================================================
// Convenience functions
// ============================================================================

/**
 * Set the global log level.
 */
inline void set_log_level(LogLevel level) {
    Logger::instance().set_level(level);
}

/**
 * Set the global log level from string.
 */
inline void set_log_level(const std::string& level_str) {
    Logger::instance().set_level(string_to_log_level(level_str));
}

/**
 * Get the current log level.
 */
inline LogLevel get_log_level() {
    return Logger::instance().get_level();
}

/**
 * Check if a log level is enabled.
 */
inline bool is_log_enabled(LogLevel level) {
    return Logger::instance().is_enabled(level);
}

/**
 * Enable or disable timestamps.
 */
inline void set_log_timestamp(bool show) {
    Logger::instance().set_show_timestamp(show);
}

/**
 * Enable or disable component tags.
 */
inline void set_log_component(bool show) {
    Logger::instance().set_show_component(show);
}

// ============================================================================
// Logging macros
// ============================================================================

// Main logging macros with component tag
#define CALIBY_LOG_DEBUG(component, ...) \
    do { \
        if (caliby::Logger::instance().is_enabled(caliby::LogLevel::DEBUG)) { \
            caliby::Logger::instance().log(caliby::LogLevel::DEBUG, component, __VA_ARGS__); \
        } \
    } while (0)

#define CALIBY_LOG_INFO(component, ...) \
    do { \
        if (caliby::Logger::instance().is_enabled(caliby::LogLevel::INFO)) { \
            caliby::Logger::instance().log(caliby::LogLevel::INFO, component, __VA_ARGS__); \
        } \
    } while (0)

#define CALIBY_LOG_WARN(component, ...) \
    do { \
        if (caliby::Logger::instance().is_enabled(caliby::LogLevel::WARN)) { \
            caliby::Logger::instance().log(caliby::LogLevel::WARN, component, __VA_ARGS__); \
        } \
    } while (0)

#define CALIBY_LOG_ERROR(component, ...) \
    do { \
        if (caliby::Logger::instance().is_enabled(caliby::LogLevel::ERROR)) { \
            caliby::Logger::instance().log(caliby::LogLevel::ERROR, component, __VA_ARGS__); \
        } \
    } while (0)

// Short-form macros (without component)
#define LOG_DEBUG(...) CALIBY_LOG_DEBUG("", __VA_ARGS__)
#define LOG_INFO(...)  CALIBY_LOG_INFO("", __VA_ARGS__)
#define LOG_WARN(...)  CALIBY_LOG_WARN("", __VA_ARGS__)
#define LOG_ERROR(...) CALIBY_LOG_ERROR("", __VA_ARGS__)

} // namespace caliby
