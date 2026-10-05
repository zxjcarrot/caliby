#include "transaction.hpp"

#include "logging.hpp"
#ifdef CALIBY_ENABLE_WAL
#include "recovery/log_entry.hpp"
#include "recovery/log_manager.hpp"
#endif

#include <climits>
#include <cstring>
#include <random>

namespace caliby {
namespace tx {

#ifdef CALIBY_ENABLE_WAL

static std::atomic<txid_t> next_txn_id{1};

static txid_t NextTxnId() { return next_txn_id.fetch_add(1); }

static bool wal_enabled() {
    return recovery::LogManager::HasInstance();
}

recovery::LogWorker& Transaction::LogWorker() const {
    return recovery::LogManager::Instance().LocalLogWorker();
}

void Transaction::StartTransaction() {
    // Sync this worker's GSN clock to the global watermark so that our worker
    // never lags behind pages written by previous workers (LeanStore
    // transaction_manager.cc:49-56). Prevents the situation where a LOCAL GSN
    // would be lower than a page's p_gsn.
    auto& worker = LogWorker();
    if (worker.w_gsn_clock < recovery::LogManager::global_sync_to_this_gsn.load(
                                 std::memory_order_acquire)) {
        worker.w_gsn_clock = recovery::LogManager::global_sync_to_this_gsn.load(
            std::memory_order_acquire);
    }
    worker.rfa_gsn_flushed = recovery::LogManager::global_min_gsn_flushed.load(
        std::memory_order_acquire);
}

#else  // !CALIBY_ENABLE_WAL

static txid_t NextTxnId() { return 0; }

#endif  // CALIBY_ENABLE_WAL

void Transaction::EnterTxn() {
    assert(!IsRunning() && "nested transactions are not supported");

    txn_id = NextTxnId();
    is_read_only = true;
    gsn_vector.clear();
    state = TxState::STARTED;

#ifdef CALIBY_ENABLE_WAL
    if (wal_enabled()) {
        auto& worker = LogWorker();
        StartTransaction();
        {
            auto& entry = worker.ReserveLogMetaEntry();            entry.type = static_cast<uint8_t>(recovery::LogEntry::Type::TX_START);
            entry.rc.w_id = worker.w_id;
            entry.rc.txn = txn_id;
            worker.SubmitActiveLogEntry();
        }
    }
#endif
}

bool Transaction::CommitTransaction() {
    if (!IsRunning()) return false;

    state = TxState::READY_TO_COMMIT;

#ifdef CALIBY_ENABLE_WAL
    if (wal_enabled()) {
        auto& worker = LogWorker();
        StartTransaction();
        const size_t payload_size = SerializedVectorSize();
        auto& entry = worker.ReserveLogCommitEntry(payload_size);
        entry.type = static_cast<uint8_t>(recovery::LogEntry::Type::TX_COMMIT);
        entry.rc.w_id = worker.w_id;
        entry.rc.txn = txn_id;
        SerializeVector(entry.payload());
        worker.SubmitActiveLogEntry();

        // Phase 4 (group commit) will make this asynchronous via a
        // precommitted queue. For now, flush + fsync synchronously so that
        // durability of a committed transaction is guaranteed on return, and
        // the buffer-manager eviction gate (p_gsn <= min_flushed_gsn) makes
        // progress deterministically.
        worker.log_buffer.LogFlush(&worker, [] {});
        recovery::fdatasync_fd(worker.log_manager->wal_fd());
        worker.log_manager->AdvanceFlushWatermark();
        is_read_only = true;
        // no-op otherwise
    }
#endif

    state = TxState::COMMITTED;
    return true;
}

void Transaction::AbortTransaction() {
    if (!IsRunning()) return;

#ifdef CALIBY_ENABLE_WAL
    if (wal_enabled()) {
        auto& worker = LogWorker();
        auto& entry = worker.ReserveLogMetaEntry();
        entry.type = static_cast<uint8_t>(recovery::LogEntry::Type::TX_ABORT);
        entry.rc.w_id = worker.w_id;
        entry.rc.txn = txn_id;
        worker.SubmitActiveLogEntry();
        worker.log_buffer.LogFlush(&worker, [] {});
    }
#endif
    gsn_vector.clear();
    state = TxState::ABORTED;
}

}  // namespace tx
}  // namespace caliby
