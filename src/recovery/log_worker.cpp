#include "recovery/log_worker.hpp"

#include "logging.hpp"
#include "recovery/log_manager.hpp"

#include <cstring>

namespace caliby {
namespace recovery {

LogWorker::LogWorker(wid_t w_id_, LogManager* mgr, std::atomic<bool>* running)
    : w_id(w_id_),
      log_manager(mgr),
      is_running(running),
      log_buffer(mgr ? mgr->buffer_size_mb * 1024ull * 1024 : 0, running) {}

LogWorker::~LogWorker() = default;

gsn_t LogWorker::GetCurrentGSN() { return w_gsn_clock; }

void LogWorker::SetCurrentGSN(gsn_t g) {
    w_gsn_clock = g;
    if (log_manager) {
        auto cur = log_manager->next_gsn_.load(std::memory_order_acquire);
        while (g > cur) {
            log_manager->next_gsn_.compare_exchange_weak(cur, g,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire);
        }
    }
}

void LogWorker::MaybeAutonomousFlush() {
    if (!log_manager) return;
    uint64_t dirty = log_buffer.wal_cursor -
                     log_buffer.write_cursor.load(std::memory_order_acquire);
    // 256 KB threshold by default. Caliby workloads are not latency-tuned
    // here so we keep this conservative.
    constexpr uint64_t kAutonomousFlushThreshold = 256ull * 1024;
    if (dirty >= kAutonomousFlushThreshold) {
        log_buffer.LogFlush(this, [] {});
    }
}

LogMetaEntry& LogWorker::ReserveLogMetaEntry() {
    auto& entry = *reinterpret_cast<LogMetaEntry*>(log_buffer.Current());
    entry.type_and_size = 0;
    entry.size = sizeof(LogMetaEntry);
    entry.chksum = LogEntry::CHKSUM_PLACEHOLDER;
    entry.rc = RecoveryInfo{};
    active_log = &entry;
    return entry;
}

TxnCommitEntry& LogWorker::ReserveLogCommitEntry(uint64_t payload_size) {
    auto& entry = *reinterpret_cast<TxnCommitEntry*>(log_buffer.Current());
    entry.type_and_size = 0;
    entry.size = static_cast<uint32_t>(sizeof(TxnCommitEntry) + payload_size);
    entry.chksum = LogEntry::CHKSUM_PLACEHOLDER;
    entry.rc = RecoveryInfo{};
    entry.vector_size = static_cast<uint32_t>(payload_size);
    active_log = &entry;
    return entry;
}

DataEntry& LogWorker::ReserveDataLog(uint64_t payload_size, pageid_t pid) {
    auto& entry = *reinterpret_cast<DataEntry*>(log_buffer.Current());
    entry.type_and_size = 0;
    entry.type = static_cast<uint8_t>(LogEntry::Type::DATA_ENTRY);
    entry.size = static_cast<uint32_t>(sizeof(DataEntry) + payload_size);
    entry.chksum = LogEntry::CHKSUM_PLACEHOLDER;
    entry.rc = RecoveryInfo{};
    entry.pid = pid;
    entry.prev_gsn = 0;
    active_log = &entry;
    return entry;
}

AllocNewPageEntry& LogWorker::ReserveAllocNewPageLogEntry(pageid_t new_page_id) {
    auto& entry = *reinterpret_cast<AllocNewPageEntry*>(log_buffer.Current());
    entry.type_and_size = 0;
    entry.type = static_cast<uint8_t>(LogEntry::Type::ALLOC_NEW_PAGE);
    entry.size = sizeof(AllocNewPageEntry);
    entry.chksum = LogEntry::CHKSUM_PLACEHOLDER;
    entry.pid = 0;
    entry.new_page_id = new_page_id;
    active_log = &entry;
    return entry;
}

FreeExtentEntry& LogWorker::ReserveFreeExtentLogEntry(bool /*to_free*/, pageid_t start_pid, pageid_t page_count) {
    auto& entry = *reinterpret_cast<FreeExtentEntry*>(log_buffer.Current());
    entry.type_and_size = 0;
    entry.type = static_cast<uint8_t>(LogEntry::Type::FREE_EXTENT);
    entry.size = sizeof(FreeExtentEntry);
    entry.chksum = LogEntry::CHKSUM_PLACEHOLDER;
    entry.rc = RecoveryInfo{};
    entry.start_pid = start_pid;
    entry.lp_size   = page_count;
    active_log = &entry;
    return entry;
}

bool LogWorker::SubmitActiveLogEntry() {
    if (!active_log) return false;
    // Stamp the size field, checksum, and advance the GSN clock.
    active_log->ComputeChksum();
    uint32_t sz = active_log->size;
    // If the entry is bigger than the current block, fall back to a new block
    // (we don't support per-block fragmentation in this simple port).
    RotateBlockIfNeeded(sz);
    log_buffer.wal_cursor += sz;
    // For DATA_ENTRY / TX_COMMIT etc. the caller has already filled `prev_gsn`
    // and we just need to stamp the GSN after the fact - but we stamp it in
    // the reservation helpers in the simple path. We update both:
    if (active_log->type == static_cast<uint8_t>(LogEntry::Type::DATA_ENTRY) ||
        active_log->type == static_cast<uint8_t>(LogEntry::Type::TX_START) ||
        active_log->type == static_cast<uint8_t>(LogEntry::Type::TX_COMMIT) ||
        active_log->type == static_cast<uint8_t>(LogEntry::Type::TX_ABORT) ||
        active_log->type == static_cast<uint8_t>(LogEntry::Type::FREE_EXTENT) ||
        active_log->type == static_cast<uint8_t>(LogEntry::Type::REUSE_EXTENT)) {
        auto* rc = reinterpret_cast<RecoveryInfo*>(reinterpret_cast<uint8_t*>(active_log) + sizeof(LogEntry));
        rc->gsn = ++w_gsn_clock;
    }
    w_state.last_wal_cursor = log_buffer.wal_cursor;
    w_state.last_gsn        = w_gsn_clock;
    active_log = nullptr;
    return true;
}

void LogWorker::RotateBlockIfNeeded(uint64_t bytes_to_write) {
    if (!log_manager) return;
    (void)log_manager->WALBlocks();
    if (!cur_block_) {
        cur_block_ = log_manager->OpenNextBlock();
    }
    // If writing these bytes would overflow the block, close it and open
    // a new one. (In practice, single records are << block size.)
    uint64_t next = cur_block_->block_next.load(std::memory_order_relaxed);
    if (next + bytes_to_write > cur_block_->block_size) {
        // Close the current block
        cur_block_->block_gsn = w_gsn_clock;
        cur_block_->state.store(static_cast<uint8_t>(WALBlock::State::CLOSED),
                                std::memory_order_release);
        // Open a new one
        cur_block_ = log_manager->OpenNextBlock();
    }
}

uint64_t LogWorker::FlushRange(uint64_t buffer_offset, uint64_t n_bytes,
                               const std::function<void()>& async_fn) {
    if (!log_manager || n_bytes == 0) return 0;
    if (!cur_block_) {
        // The worker has no block yet - allocate one. This happens on the
        // first flush of a session.
        cur_block_ = log_manager->OpenNextBlock();
    }
    // Buffer layout: wal_buffer is a single mmap region that the worker
    // fills linearly. We write the dirty range [buffer_offset, buffer_offset
    // + n_bytes) to the WAL file at the current block's position.
    //
    // We honor per-block boundaries: if the dirty range would overflow the
    // current block, we close this block and let the next flush pick a new
    // one. For the simple test path we assume the buffer fits inside one
    // block (the default 1MB buffer is much smaller than the default 1MB
    // block size, so a flush typically lands inside one block).
    uint64_t block_base = cur_block_->block_base;
    uint64_t block_size = cur_block_->block_size;
    uint64_t block_next = cur_block_->block_next.load(std::memory_order_acquire);
    uint64_t avail = block_size - block_next;
    uint64_t to_write = std::min(n_bytes, avail);
    if (to_write == 0) {
        if (async_fn) async_fn();
        return 0;
    }
    uint64_t physical_off = log_manager->wal_region_start() + block_base + block_next;
    ssize_t n = pwrite_all(log_manager->wal_fd(),
                           log_buffer.wal_buffer + buffer_offset,
                           to_write, physical_off);
    if (n < 0) {
        CALIBY_LOG_ERROR("LogWorker", "pwrite failed: offset=", physical_off,
                         " size=", to_write);
        return 0;
    }
    cur_block_->block_next.fetch_add(static_cast<uint64_t>(n), std::memory_order_acq_rel);
    if (async_fn) async_fn();
    return static_cast<uint64_t>(n);
}

}  // namespace recovery
}  // namespace caliby
