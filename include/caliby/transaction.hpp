#pragma once

// Transaction abstraction for Caliby, ported from LeanStore's transaction::Transaction.
//
// One *active* transaction per thread (TLS). Every write operation must run
// inside a transaction; page guards (GuardX) refuse to emit WAL records
// without an active write transaction.
//
// In Caliby, user operations (addPoint, insert, ...) run in auto-commit mode:
// TransactionAutoCommitScope opens a transaction in its ctor and commits in
// its dtor. Explicit multi-statement transactions can be added later without
// changing the guard API. Nested transactions are NOT supported.

#include "recovery/log_worker.hpp"  // recovery::{gsn_t,txid_t,wid_t, LogWorker}

#include <atomic>
#include <cassert>
#include <cstring>
#include <unordered_map>

namespace caliby {
namespace tx {

using recovery::gsn_t;
using recovery::txid_t;
using recovery::wid_t;

enum class TxState : uint8_t {
    STARTED = 0,
    READY_TO_COMMIT = 1,  // all log work done (group commit consumes in Phase 4)
    ABORTED = 2,
    COMMITTED = 3,
};

struct Transaction {
    txid_t   txn_id = 0;
    TxState  state = TxState::ABORTED;  // idle until EnterTxn
    bool     is_read_only = true;

    // Cross-worker dependency vector: gsn_vector[wid] = highest GSN written by
    // worker `wid` onto pages this transaction touched (fed by
    // GuardX::DetectGSNDependency, Phase 3). Serialized into TX_COMMIT so
    // recovery can classify winner/loser transactions across workers.
    std::unordered_map<wid_t, gsn_t> gsn_vector;

    // ---- Identity & state ---------------------------------------------------
    bool   IsRunning() const { return state == TxState::STARTED || state == TxState::READY_TO_COMMIT; }
    bool   IsReadOnly() const { return is_read_only; }
    txid_t TxnId() const { return txn_id; }
    TxState GetState() const { return state; }

    // ---- Lifecycle -----------------------------------------------------------
    // Enter a transaction on this thread (asserts none is running).
    void EnterTxn();

    // Commit: emits TX_COMMIT with the serialized dependency vector and makes
    // the log durable (synchronous flush + fsync; group commit is Phase 4).
    // Returns true if the transaction was running and got committed.
    bool CommitTransaction();

    // Abort: emits TX_ABORT.
    void AbortTransaction();

    void MarkAsWrite() { is_read_only = false; }

    // ---- Dependency vector ---------------------------------------------------
    void UpdateDependencyVector(wid_t owner, gsn_t gsn) {
        auto it = gsn_vector.find(owner);
        if (it == gsn_vector.end() || it->second < gsn) {
            gsn_vector[owner] = gsn;
        }
    }
    const std::unordered_map<wid_t, gsn_t>& GetGSNVector() const { return gsn_vector; }

    // Payload layout: [u32 num_entries][(u32 wid, u64 gsn) * num_entries]
    size_t SerializedVectorSize() const {
        return sizeof(uint32_t) + gsn_vector.size() * (sizeof(uint32_t) + sizeof(gsn_t));
    }
    void SerializeVector(uint8_t* dest) const {
        uint32_t count = static_cast<uint32_t>(gsn_vector.size());
        std::memcpy(dest, &count, sizeof(count));
        uint8_t* cursor = dest + sizeof(count);
        for (auto& [wid, gsn] : gsn_vector) {  // unordered ok: used for winner/loser only
            uint32_t w = wid;
            gsn_t g = gsn;
            std::memcpy(cursor, &w, sizeof(w));
            std::memcpy(cursor + sizeof(w), &g, sizeof(g));
            cursor += sizeof(w) + sizeof(g);
        }
    }

    // ---- Logging --------------------------------------------------------------
    // Per-thread LogWorker (requires LogManager to be running).
    recovery::LogWorker& LogWorker() const;

    // Start/resume: raise the worker GSN clock to the global watermark so this
    // worker never lags behind pages written by other workers
    // (LeanStore transaction_manager.cc:49-56).
    void StartTransaction();
};

// The thread-local active transaction (namespace-scope inline thread_local,
// C++17: one instance shared across TUs).
inline thread_local Transaction tl_active_txn;
inline Transaction& active_txn() { return tl_active_txn; }

// RAII auto-commit transaction scope.
//
//   {
//       TransactionAutoCommitScope txn;  // EnterTxn
//       ... page mutations under guards ...
//   }                                    // -> CommitTransaction
struct TransactionAutoCommitScope {
    TransactionAutoCommitScope() { tx::active_txn().EnterTxn(); }
    ~TransactionAutoCommitScope() {
        auto& txn = tx::active_txn();
        if (txn.IsRunning()) {
            txn.CommitTransaction();
        }
    }
    TransactionAutoCommitScope(const TransactionAutoCommitScope&) = delete;
    TransactionAutoCommitScope& operator=(const TransactionAutoCommitScope&) = delete;
};

}  // namespace tx
}  // namespace caliby
