#pragma once

#include <string>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>

// Forward declarations for libssh2 types
struct _LIBSSH2_SESSION;
struct _LIBSSH2_SFTP;
typedef struct _LIBSSH2_SESSION LIBSSH2_SESSION;
typedef struct _LIBSSH2_SFTP LIBSSH2_SFTP;

// File transfer request from main thread to SFTP worker thread
struct FileTransferRequest {
    std::string local_path;
    std::string remote_path;
    size_t file_size;
};

// File transfer progress updated by worker thread, read by main thread
struct FileTransferProgress {
    std::string filename;
    size_t bytes_transferred = 0;
    size_t total_bytes = 0;
    float progress_percent = 0.0f;

    enum State { IDLE, CONNECTING, TRANSFERRING, COMPLETED, FAILED };
    State state = IDLE;

    std::string error_message;
    std::chrono::steady_clock::time_point completion_time;
};

// SFTP client component - mirrors VNCClient threading pattern
struct SFTPClient {
    // SSH/SFTP session handles
    int sock = -1;
    LIBSSH2_SESSION* session = nullptr;
    LIBSSH2_SFTP* sftp_session = nullptr;

    // Connection info (derived from VNCClient host)
    std::string host;
    int port = 22;
    std::string username = "grok";
    std::string password = "grok";  // Simple password auth

    // Connection state
    enum ConnectionState { DISCONNECTED, CONNECTING, CONNECTED, ERROR };
    std::atomic<ConnectionState> conn_state{DISCONNECTED};

    // Transfer queue and thread (mirroring VNCClient pattern)
    std::deque<FileTransferRequest> transfer_queue;
    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::thread worker_thread;
    std::atomic<bool> thread_running{false};
    std::atomic<bool> thread_should_stop{false};

    // Current transfer progress (atomic updates from worker thread)
    FileTransferProgress current_progress;
    mutable std::mutex progress_mutex;

    // Delete copy constructor and copy assignment (non-copyable due to thread/mutex/atomic)
    SFTPClient(const SFTPClient&) = delete;
    SFTPClient& operator=(const SFTPClient&) = delete;

    // Default constructor
    SFTPClient() = default;

    // Destructor - ensure thread is properly joined
    ~SFTPClient() {
        if (thread_running) {
            thread_should_stop = true;
            queue_cv.notify_all();
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
    }

    // Move constructor - CRITICAL: This should never be called after thread starts!
    SFTPClient(SFTPClient&& other) noexcept
        : sock(other.sock), session(other.session), sftp_session(other.sftp_session),
          host(std::move(other.host)), port(other.port),
          username(std::move(other.username)),
          password(std::move(other.password)),
          conn_state(other.conn_state.load()),
          transfer_queue(std::move(other.transfer_queue)),
          queue_mutex(), queue_cv(),
          worker_thread(std::move(other.worker_thread)),
          thread_running(other.thread_running.load()),
          thread_should_stop(other.thread_should_stop.load()),
          current_progress(other.current_progress),
          progress_mutex()
    {
        std::cerr << "[SFTP MOVE] SFTPClient moved from " << (void*)&other << " to " << (void*)this << std::endl;

        // WARNING: If a thread is running, moving is unsafe!
        if (other.thread_running) {
            std::cerr << "FATAL: SFTPClient moved while thread is running!" << std::endl;
            std::abort();
        }

        other.sock = -1;
        other.session = nullptr;
        other.sftp_session = nullptr;
    }

    // Move assignment
    SFTPClient& operator=(SFTPClient&& other) noexcept {
        if (this != &other) {
            sock = other.sock;
            session = other.session;
            sftp_session = other.sftp_session;
            host = std::move(other.host);
            port = other.port;
            username = std::move(other.username);
            password = std::move(other.password);
            conn_state.store(other.conn_state.load());
            transfer_queue = std::move(other.transfer_queue);
            worker_thread = std::move(other.worker_thread);
            thread_running.store(other.thread_running.load());
            thread_should_stop.store(other.thread_should_stop.load());
            current_progress = other.current_progress;

            other.sock = -1;
            other.session = nullptr;
            other.sftp_session = nullptr;
        }
        return *this;
    }
};

// Tag component for entities with active SFTP transfers (for rendering system)
struct HasSFTPTransfer {};
