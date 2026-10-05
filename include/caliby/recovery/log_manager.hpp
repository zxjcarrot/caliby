#pragma once

#include "log_buffer.hpp"
#include "log_entry.hpp"
#include "log_io_segment.hpp"
#include "log_worker.hpp"
#include "wal_block_manager.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace caliby {
namespace recovery {

class Checkpointer;
class RecoveryManager;
class GroupCommitExecutor;

// Central WAL coordinator. Owns:
//   - the WAL file descriptor
//   - the WALBlockManager (logical block ring)
//   - per-worker LogWorker contexts
//   - the autonomous group commit / checkpointer threads
//
// Layout on disk:
//   [ MasterRecordEntry (one 4KB block at offset 0) ]
//   [ Block 0 | Block 1 | ... | Block N-1 ]   // each is wal_block_size bytes
class LogManager {
public:
    // Tunables - read from env at construction time.
    uint64_t wal_block_size = 1ull * 1024 * 1024;   // 1 MB
    uint64_t wal_size_mb    = 64;                    // default 64 MB
    uint32_t num_workers    = 1;
    uint64_t buffer_size_mb = 4;                     // per-worker circular log buffer

    static std::atomic<gsn_t> global_min_gsn_flushed;
    static std::atomic<gsn_t> global_sync_to_this_gsn;

private:
    static inline LogManager* s_instance_ = nullptr;

public:

    explicit LogManager(std::atomic<bool>& is_running, const std::string& wal_path = "");
    ~LogManager();

    // Singleton accessor. The Caliby buffer manager owns one LogManager for
    // its lifetime. Returns a reference to the process-wide instance.
    static LogManager& Instance();
    static void SetInstance(LogManager* mgr);
    static bool HasInstance() { return s_instance_ != nullptr; }

    // Init / shutdown.
    void StartWorkers();
    void Shutdown();

    // Per-thread handle.
    LogWorker& LocalLogWorker();

    // The WAL file descriptor. Used by LogWorker::FlushRange.
    int  wal_fd() const { return wal_fd_; }
    std::string wal_path() const { return wal_path_; }
    uint64_t wal_region_start() const { return wal_region_start_; }   // = 4KB (master record)
    uint64_t wal_region_end() const { return wal_region_end_; }

    WALBlockManager& WALBlocks() { return *wal_blocks_; }
    const WALBlockManager& WALBlocks() const { return *wal_blocks_; }

    // Persist a fresh MasterRecordEntry to the WAL file. Should be called on
    // a clean open (no prior session) or after a successful recovery.
    void WriteMasterRecord();

    // Read the previous session's MasterRecordEntry. Returns false if none
    // found or if validation fails.
    bool ReadMasterRecord(MasterRecordEntry& out);

    // Trigger an autonomous flush + GSN-watermark advance. Returns the new
    // min-flushed GSN.
    gsn_t AdvanceFlushWatermark();

    // Start background threads (checkpointer, optional group committer).
    void StartBackgroundThreads();
    void StopBackgroundThreads();

    // The checkpointer is consulted by recovery during Bootstrap.
    Checkpointer* checkpointer() { return checkpointer_.get(); }

private:
    friend class LogWorker;
    friend class Checkpointer;
    friend class RecoveryManager;
    friend class GroupCommitExecutor;

    int                      wal_fd_ = -1;
    std::atomic<bool>*       is_running_ = nullptr;
    std::string              wal_path_;
    uint64_t                 wal_region_start_ = 0;
    uint64_t                 wal_region_end_   = 0;

    std::unique_ptr<WALBlockManager> wal_blocks_;
    std::vector<std::unique_ptr<LogWorker>> workers_;
    std::vector<WorkerConsistentState>     w_state_;
    std::vector<WorkerConsistentState>     commit_state_;

    // GSN clock for issuing new GSNs.
    std::atomic<gsn_t> next_gsn_{1};

    // Background threads.
    std::unique_ptr<Checkpointer>        checkpointer_;
    // std::unique_ptr<GroupCommitExecutor> group_committer_;  // not used (WILO)

    // Open the WAL file (creates if missing, truncates if no master record).
    void OpenWalFile();

    // Allocate a fresh WAL block via the WALBlockManager and set it on the
    // worker. May grow the WAL file by ftruncate (one-time during the
    // session's lifetime).
    WALBlock* OpenNextBlock();
};

// ---- File-IO helpers --------------------------------------------------
// Used by LogWorker::FlushRange and by RecoveryManager. Always synchronous
// pwrite + optional fdatasync; the LogBackend is intentionally simple for
// Caliby.
ssize_t pwrite_all(int fd, const void* buf, size_t n_bytes, off_t offset);
int    fdatasync_fd(int fd);

}  // namespace recovery
}  // namespace caliby
