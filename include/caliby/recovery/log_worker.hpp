#pragma once

#include "log_buffer.hpp"
#include "log_entry.hpp"
#include "log_io_segment.hpp"
#include "wal_block_manager.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace caliby {
namespace recovery {

class LogManager;

// Snapshot of a worker's consistent state - what its durable frontier is.
struct WorkerConsistentState {
    uint64_t last_wal_cursor   = 0;
    gsn_t    last_gsn          = 0;
    gsn_t    precommitted_tx_commit_ts = 0;
};

// Per-worker logging context. Owns a circular LogBuffer and an io backend.
// There is one LogWorker per "logical worker" (in Caliby we use 1 worker for
// now, matching the existing single-threaded buffer manager; multi-worker
// support is straightforward to add later).
struct LogWorker {
    // Identity (set by LogManager).
    wid_t     w_id           = 0;
    LogManager* log_manager = nullptr;
    std::atomic<bool>* is_running = nullptr;

    // Active log entry being assembled. Points into log_buffer.wal_buffer.
    LogEntry* active_log = nullptr;

    // Worker GSN clock - local copy; advanced on every reservation. Synced
    // from LogManager's global GSN on demand.
    gsn_t     w_gsn_clock   = 1;   // start at 1 so 0 means "no record"
    gsn_t     rfa_gsn_flushed = 0; // cached "global min flushed" for RFA
    gsn_t     last_unharden_commit_ts = 0;

    // Cached consistent state for the checkpointer / group committer.
    WorkerConsistentState w_state;

    // Buffer and IO.
    LogBuffer      log_buffer;
    LogIOSegment   write_blk;  // scratch segment for Serialization

    LogWorker() = default;
    LogWorker(wid_t w_id, LogManager* mgr, std::atomic<bool>* running);
    ~LogWorker();

    // GSN helpers.
    gsn_t GetCurrentGSN();
    void  SetCurrentGSN(gsn_t g);

    // Atomically bump the local GSN clock and return the new value.
    gsn_t AdvanceAndGetGSN() { return ++w_gsn_clock; }

    // If the dirty log exceeds the worker write batch size, trigger an
    // autonomous flush (WILO variant). No-op otherwise.
    void MaybeAutonomousFlush();

    // ---- Reservation API --------------------------------------------------
    // Caller assembles a log entry by setting fields directly on the returned
    // reference, then calls SubmitActiveLogEntry().
    LogMetaEntry&   ReserveLogMetaEntry();
    TxnCommitEntry& ReserveLogCommitEntry(uint64_t payload_size);
    DataEntry&      ReserveDataLog(uint64_t payload_size, pageid_t pid);
    AllocNewPageEntry& ReserveAllocNewPageLogEntry(pageid_t new_page_id);
    FreeExtentEntry&   ReserveFreeExtentLogEntry(bool to_free, pageid_t start_pid, pageid_t page_count);

    // Finalize the active entry: stamp chksum, advance wal_cursor.
    bool SubmitActiveLogEntry();

    // ---- Buffer flush / IO -----------------------------------------------
    // Flush a contiguous range of bytes from the worker's LogBuffer to disk
    // via the LogManager (which owns the wal_fd_).
    // Returns the number of bytes actually written.
    uint64_t FlushRange(uint64_t buffer_offset, uint64_t n_bytes,
                        const std::function<void()>& async_fn);

    // Optionally open the next WAL block (or grow the WAL if necessary).
    // Called when the active block is full.
    void RotateBlockIfNeeded(uint64_t bytes_to_write);

    // Current block we're writing into. May be null before the first write.
    WALBlock* cur_block_ = nullptr;
};

}  // namespace recovery
}  // namespace caliby
