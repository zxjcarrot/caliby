#pragma once

#include "log_entry.hpp"
#include "log_io_segment.hpp"
#include "log_manager.hpp"
#include "recovery_backend.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace caliby {
namespace recovery {

class RecoveryManager {
public:
    explicit RecoveryManager(LogManager* log_manager);
    ~RecoveryManager();

    // Step 1: read the previous session's master record and configure state.
    void RecoveryPreparation();

    // Step 2: read every WAL block and bucket DataEntry records per page.
    uint64_t MaterializeLogs();

    // Step 3: register the materialized blocks with the WALBlockManager.
    void ApplyRecoveredBlocks();

    // Step 4: classify each transaction as winner or loser using the GSN
    // vector from TX_COMMIT records.
    void Analysis();

    // Step 5: redo all winner-txn page modifications.
    void Redo();

    // Step 5b: redo free-extent records.
    void RedoFreeExtents();

    // Step 6: undo all loser-txn page modifications (CLRs).
    void Undo();

    // Number of times we have observed a previously-allocated PID.
    std::atomic<pageid_t> max_logged_pid_{0};

private:
    LogManager* log_manager_ = nullptr;
    std::unique_ptr<RecoveryBackend> backend_;

    // Per-worker, per-page log buffers. BTree redo iterates in GSN order
    // over the (wid, pid) bucket.
    std::unordered_map<pageid_t, std::vector<LogIOSegment::Buffer>> page_log_[MAX_NUMBER_OF_WORKERS];

    // Per-worker free-extent logs.
    std::vector<LogIOSegment::Buffer> free_extent_logs_[MAX_NUMBER_OF_WORKERS];

    // Per-worker (block_idx, block_max_gsn) for the WALBlockManager.
    std::vector<std::pair<uint32_t, gsn_t>> recovered_blocks_[MAX_NUMBER_OF_WORKERS];

    // Maximum allocated page ID per worker (for redo).
    std::atomic<pageid_t> max_alloc_cnt_[MAX_NUMBER_OF_WORKERS];

    // Winner / loser sets (per-worker; merged at the end of Analysis).
    std::unordered_set<txid_t> tl_losers_[MAX_NUMBER_OF_WORKERS];
    std::unordered_map<txid_t, std::unordered_map<wid_t, gsn_t>> tl_winners_[MAX_NUMBER_OF_WORKERS];

    // Per-PID "have we redone this page" flag.
    std::unique_ptr<std::atomic<bool>[]> has_recovered_;

    // 1-page redo counter (work-stealing for parallel Redo()).
    std::atomic<pageid_t> redo_pid_counter_{0};
    // GSN frontier per worker.
    std::atomic<gsn_t> global_durable_gsn_[MAX_NUMBER_OF_WORKERS];

    MasterRecordEntry prev_session_{};
    bool has_prev_session_ = false;

    // Internal helpers.
    void AnalyzeLogSegment(const LogIOSegment::Buffer& buf, gsn_t& block_max_gsn);
    void RedoLog(const DataEntry* log, uint8_t* page_ptr);
    bool IsWinner(txid_t txn) const;
    bool IsLoser(txid_t txn) const;
};

}  // namespace recovery
}  // namespace caliby
