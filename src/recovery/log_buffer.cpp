#include "recovery/log_buffer.hpp"

#include "logging.hpp"
#include "recovery/log_worker.hpp"

#include <sys/mman.h>
#include <cstring>

namespace caliby {
namespace recovery {

LogBuffer::LogBuffer(uint64_t buffer_size_, std::atomic<bool>* is_running_)
    : buffer_size(buffer_size_), is_running(is_running_) {
    // 4KB-aligned mmap so we can use pwrite() directly on a slice of the
    // log file.
    void* p = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        CALIBY_LOG_ERROR("LogBuffer", "mmap of size ", buffer_size, " failed");
        wal_buffer = nullptr;
        return;
    }
    wal_buffer = reinterpret_cast<uint8_t*>(p);
}

LogBuffer::~LogBuffer() {
    if (wal_buffer) {
        munmap(wal_buffer, buffer_size);
        wal_buffer = nullptr;
    }
}

uint64_t LogBuffer::TotalFreeSpace() {
    uint64_t wc = write_cursor.load(std::memory_order_acquire);
    return buffer_size - (wal_cursor - wc);
}

uint64_t LogBuffer::ContiguousFreeSpaceForNewEntry() {
    uint64_t wc = write_cursor.load(std::memory_order_acquire);
    return buffer_size - wal_cursor;
}

void LogBuffer::EnsureEnoughSpace(LogWorker* owner, uint64_t requested_size) {
    // WILO: flush our own dirty log to disk autonomously. The owner writes
    // to the LogBackend (pwrite + fdatasync). Then we reset the cursor and
    // write_cursor to 0 (the buffer is logically circular).
    //
    // We do NOT need to "consume" an entire block here -- a single LogFlush
    // drains only the dirty portion. If the block is full we CLOSE it as
    // part of the flush.
    for (;;) {
        if (TotalFreeSpace() >= requested_size) return;
        if (is_running && !is_running->load(std::memory_order_acquire)) {
            // Shutting down: give up and let the caller observe a fatal error.
            return;
        }
        // Autonomous flush
        LogFlush(owner, [] {});
        // Note: LogFlush moves write_cursor up; TotalFreeSpace will grow.
    }
}

uint64_t LogBuffer::LogFlush(LogWorker* worker, const std::function<void()>& async_fn) {
    if (!worker) return 0;
    uint64_t dirty_start = write_cursor.load(std::memory_order_acquire);
    uint64_t dirty_end   = wal_cursor;
    if (dirty_end == dirty_start) return 0;
    uint64_t dirty_bytes = dirty_end - dirty_start;
    uint64_t written = worker->FlushRange(dirty_start, dirty_bytes, async_fn);
    write_cursor.fetch_add(written, std::memory_order_acq_rel);
    // If we drained everything, reset cursors to 0 to free space for new writes.
    // (WILO: the buffer is a per-worker scratch; once it is on disk we no
    // longer need to keep it resident.)
    uint64_t wc_after = write_cursor.load(std::memory_order_acquire);
    if (wc_after == wal_cursor) {
        write_cursor.store(0, std::memory_order_release);
        wal_cursor = 0;
    }
    return written;
}

}  // namespace recovery
}  // namespace caliby
