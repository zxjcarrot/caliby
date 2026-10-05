#include "recovery/recovery_manager.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cstring>

namespace caliby {
namespace recovery {

RecoveryManager::RecoveryManager(LogManager* log_manager)
    : log_manager_(log_manager) {
    backend_ = std::make_unique<RecoveryBackend>(log_manager);
    for (uint32_t i = 0; i < MAX_NUMBER_OF_WORKERS; ++i) {
        max_alloc_cnt_[i].store(0, std::memory_order_relaxed);
        global_durable_gsn_[i].store(0, std::memory_order_relaxed);
    }
    max_logged_pid_.store(0, std::memory_order_relaxed);
    has_recovered_ = nullptr;  // allocated in PrepareRedoEnv() (Phase 4)
}

RecoveryManager::~RecoveryManager() = default;

void RecoveryManager::RecoveryPreparation() {
    if (!log_manager_) return;
    backend_->Connect();
    has_prev_session_ = backend_->RetrieveMasterRecord(prev_session_);
    if (has_prev_session_) {
        CALIBY_LOG_INFO("RecoveryManager", "Previous session: workers=",
                        backend_->PrevWorkerCount(),
                        " wal_size_mb=", backend_->PrevWalSizeMB(),
                        " block_size_mb=", backend_->PrevWalBlockSizeMB(),
                        " signature=0x", std::hex, backend_->Signature(), std::dec);
    } else {
        CALIBY_LOG_INFO("RecoveryManager", "No prior session found");
    }
}

uint64_t RecoveryManager::MaterializeLogs() {
    if (!has_prev_session_) return 0;
    uint32_t required_sig = backend_->Signature();
    uint64_t total = 0;
    backend_->ReadLogBlocks([&](uint8_t* buf, uint64_t len, uint32_t sig, uint64_t offset) {
        if (sig != required_sig && sig != 0) {
            // Signature mismatch - skip this block (it might be from an
            // earlier, abandoned session).
            return;
        }
        LogIOSegment::Buffer tmp(buf, len);
        gsn_t block_max_gsn = 0;
        AnalyzeLogSegment(tmp, block_max_gsn);
        // Find which logical block this is
        uint64_t block_idx = (offset - log_manager_->wal_region_start()) /
                             backend_->WalBlockSize();
        if (block_idx < MAX_NUMBER_OF_WORKERS) {
            recovered_blocks_[block_idx].emplace_back(
                static_cast<uint32_t>(block_idx), block_max_gsn);
        }
        ++total;
    });
    return total;
}

void RecoveryManager::ApplyRecoveredBlocks() {
    if (!log_manager_) return;
    for (uint32_t w = 0; w < MAX_NUMBER_OF_WORKERS; ++w) {
        for (auto& [idx, gsn] : recovered_blocks_[w]) {
            if (gsn > 0) {
                log_manager_->WALBlocks().RecoverBlock(idx, gsn);
            }
        }
    }
}

void RecoveryManager::AnalyzeLogSegment(const LogIOSegment::Buffer& buf, gsn_t& block_max_gsn) {
    buf.Iterate([&](const LogEntry* e) {
        auto t = static_cast<LogEntry::Type>(e->type);
        switch (t) {
            case LogEntry::Type::TX_START: {
                // Tentative loser; will be removed if TX_COMMIT shows up.
                auto* meta = reinterpret_cast<const LogMetaEntry*>(e);
                tl_losers_[meta->rc.w_id].insert(meta->rc.txn);
                break;
            }
            case LogEntry::Type::TX_COMMIT: {
                auto* tc = reinterpret_cast<const TxnCommitEntry*>(e);
                tl_losers_[tc->rc.w_id].erase(tc->rc.txn);
                break;
            }
            case LogEntry::Type::DATA_ENTRY: {
                auto* dt = reinterpret_cast<const DataEntry*>(e);
                page_log_[dt->rc.w_id][dt->pid].emplace_back(buf.data + 0, 0);
                // We need a copy of the bytes for this entry - but the
                // simple approach is to take a Buffer slice. We can't
                // take sub-Buffers from a stolen buffer easily, so we
                // walk again below. The block_max_gsn tracking happens
                // here.
                if (dt->rc.gsn > block_max_gsn) block_max_gsn = dt->rc.gsn;
                if (dt->pid > max_logged_pid_.load()) {
                    max_logged_pid_.store(dt->pid, std::memory_order_release);
                }
                break;
            }
            case LogEntry::Type::FREE_EXTENT:
            case LogEntry::Type::REUSE_EXTENT: {
                auto* fe = reinterpret_cast<const FreeExtentEntry*>(e);
                free_extent_logs_[fe->rc.w_id].emplace_back(buf.data + 0, 0);
                break;
            }
            default: break;
        }
        return true;
    });
}

void RecoveryManager::Analysis() {
    // Stub - full GSN-vector analysis lives in Phase 4.
}

void RecoveryManager::RedoFreeExtents() {}
void RecoveryManager::Redo() {}
void RecoveryManager::Undo() {}

bool RecoveryManager::IsWinner(txid_t) const { return true; }
bool RecoveryManager::IsLoser(txid_t) const { return false; }

}  // namespace recovery
}  // namespace caliby
