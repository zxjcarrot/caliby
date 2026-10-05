#pragma once

#include "log_entry.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

namespace caliby {
namespace recovery {

class LogWorker;

// Per-worker circular WAL buffer. Backed by an anonymous mmap of the requested
// size. Workers append at `wal_cursor`; the log backend advances `write_cursor`
// (a.k.a. flushed frontier) after each successful write.
struct LogBuffer {
    // The smallest record we ever emit. We use it for CARRIAGE_RETURN padding.
    static constexpr uint64_t CR_ENTRY_SIZE = sizeof(LogEntry);

    uint64_t          wal_cursor = 0;                 // next append position
    std::atomic<uint64_t> write_cursor{0};            // durable frontier
    uint8_t*          wal_buffer = nullptr;           // backing storage
    uint64_t          buffer_size = 0;                // capacity in bytes
    std::atomic<bool>* is_running = nullptr;

    LogBuffer() = default;
    LogBuffer(uint64_t buffer_size_, std::atomic<bool>* is_running_);
    ~LogBuffer();

    // No copy/move
    LogBuffer(const LogBuffer&) = delete;
    LogBuffer& operator=(const LogBuffer&) = delete;

    inline uint8_t* Current() { return wal_buffer + wal_cursor; }

    static inline uint64_t DirtyLogSize(uint64_t wal_cursor, uint64_t write_cursor) {
        return wal_cursor - write_cursor;
    }

    // Total free space in the ring (= buffer_size - dirty).
    uint64_t TotalFreeSpace();

    // Free space at the end of the current segment (limited by the current
    // block boundary). If the writer is past the current block, callers should
    // rotate to a new block before calling ContiguousFreeSpaceForNewEntry.
    uint64_t ContiguousFreeSpaceForNewEntry();

    // Spin until `requested_size` bytes are available. May flush dirty bytes
    // to disk autonomously (WILO variant).
    void EnsureEnoughSpace(LogWorker* owner, uint64_t requested_size);

    // Flush dirty bytes [write_cursor, wal_cursor) to the worker's LogBackend.
    // `async_fn` runs once the data is on disk.
    uint64_t LogFlush(LogWorker* worker, const std::function<void()>& async_fn = {});
};

}  // namespace recovery
}  // namespace caliby
