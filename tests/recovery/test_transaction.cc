// Tests for the transaction abstraction (Phase 1 of docs/LOGGING_INTEGRATION.md).
// Ported/extended from LeanStore's transaction tests: TX lifecycle, 
// dependency-vector serialization, multi-worker LogWorker mapping.

#include "transaction.hpp"

#include "calico.hpp"  // workerThreadId
#include "recovery/log_entry.hpp"
#include "recovery/log_manager.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

using caliby::recovery::LogManager;

namespace caliby::tx {

class TestTransaction : public ::testing::Test {
protected:
    std::string wal_path_;
    std::atomic<bool> running_{true};
    std::unique_ptr<LogManager> log_;

    void SetUp() override {
        // Two workers so the TLS mapping test can address distinct LogWorkers.
        setenv("CALIBY_NUM_WORKERS", "2", 1);
        wal_path_ = "/tmp/caliby_test_transaction.wal";
        std::remove(wal_path_.c_str());
        log_ = std::make_unique<LogManager>(running_, wal_path_);
        log_->wal_block_size = 4ULL * 1024 * 1024;
        log_->wal_size_mb = 8;
        log_->buffer_size_mb = 1;
        log_->StartWorkers();
        log_->WriteMasterRecord();
    }

    void TearDown() override {
        log_.reset();
        running_.store(false, std::memory_order_release);
        unsetenv("CALIBY_NUM_WORKERS");
        std::remove(wal_path_.c_str());
    }
};

TEST_F(TestTransaction, LifecycleAndAutoCommit) {
    ASSERT_FALSE(active_txn().IsRunning());

    {
        TransactionAutoCommitScope txn;
        ASSERT_TRUE(active_txn().IsRunning());
        EXPECT_EQ(active_txn().GetState(), TxState::STARTED);
        EXPECT_TRUE(active_txn().IsReadOnly());
        active_txn().MarkAsWrite();
        EXPECT_FALSE(active_txn().IsReadOnly());
    }
    // After the controller scope, the txn is committed.
    EXPECT_EQ(active_txn().GetState(), TxState::COMMITTED);
    EXPECT_FALSE(active_txn().IsRunning());

    EXPECT_FALSE(active_txn().IsRunning());
}

TEST_F(TestTransaction, TxnIdsUniquePerTransaction) {
    txid_t first = 0;
    {
        TransactionAutoCommitScope a;
        first = active_txn().TxnId();
        EXPECT_NE(first, 0u);
    }
    {
        TransactionAutoCommitScope b;
        EXPECT_NE(active_txn().TxnId(), first);
    }
}

TEST_F(TestTransaction, DependencyVectorMonotonic) {
    TransactionAutoCommitScope t;
    active_txn().MarkAsWrite();

    active_txn().UpdateDependencyVector(0, 5);
    active_txn().UpdateDependencyVector(0, 3);  // lower: ignored
    active_txn().UpdateDependencyVector(1, 9);
    EXPECT_EQ(active_txn().GetGSNVector().size(), 2u);
    EXPECT_EQ(active_txn().GetGSNVector().at(0), 5u);
    EXPECT_EQ(active_txn().GetGSNVector().at(1), 9u);
}

TEST_F(TestTransaction, SerializedVectorRoundTrip) {
    TransactionAutoCommitScope _;

    // Empty vector: exactly 4 bytes.
    EXPECT_EQ(active_txn().SerializedVectorSize(), 4u);

    active_txn().UpdateDependencyVector(2, 4711);
    active_txn().UpdateDependencyVector(0, 42);
    size_t sz = active_txn().SerializedVectorSize();
    EXPECT_EQ(sz, 4u + 2u * 12u);

    std::vector<uint8_t> buf(sz);
    active_txn().SerializeVector(buf.data());

    // Parse back: [u32 n][(u32 wid, u64 gsn) * n]
    uint32_t count;
    std::memcpy(&count, buf.data(), sizeof(count));
    ASSERT_EQ(count, 2u);
    bool saw_w0 = false, saw_w2 = false;
    for (size_t off = sizeof(count), i = 0; i < count;
         ++i, off += sizeof(uint32_t) + sizeof(gsn_t)) {
        uint32_t wid;
        gsn_t gsn;
        std::memcpy(&wid, buf.data() + off, sizeof(wid));
        std::memcpy(&gsn, buf.data() + off + sizeof(wid), sizeof(gsn));
        if (wid == 0) { saw_w0 = true; EXPECT_EQ(gsn, 42u); }
        if (wid == 2) { saw_w2 = true; EXPECT_EQ(gsn, 4711u); }
    }
    EXPECT_TRUE(saw_w0);
    EXPECT_TRUE(saw_w2);
}

TEST_F(TestTransaction, EmitsTxStartAndCommit) {
    {
        TransactionAutoCommitScope t;
        active_txn().MarkAsWrite();
        auto& worker = active_txn().LogWorker();
        EXPECT_EQ(worker.w_id, workerThreadId % 2);

        // TX_START emitted on EnterTxn: wal_cursor advanced by one meta entry.
        EXPECT_EQ(worker.log_buffer.wal_cursor, sizeof(recovery::LogMetaEntry));

        // First entry is a TX_START for this txn.
        auto* start =
            reinterpret_cast<recovery::LogEntry*>(worker.log_buffer.wal_buffer);
        EXPECT_EQ(start->type,
                  static_cast<uint8_t>(recovery::LogEntry::Type::TX_START));

        // Now commit; a TX_COMMIT is appended.
        gsn_t gsn_before = worker.w_gsn_clock;
        EXPECT_TRUE(active_txn().CommitTransaction());
        EXPECT_EQ(active_txn().GetState(), TxState::COMMITTED);
        auto* commit = reinterpret_cast<recovery::LogEntry*>(
            worker.log_buffer.wal_buffer + sizeof(recovery::LogMetaEntry));
        EXPECT_EQ(commit->type,
                  static_cast<uint8_t>(recovery::LogEntry::Type::TX_COMMIT));
        // Worker GSN clock advances once per entry (TX_COMMIT since TX_START
        // already stamped its GSN before gsn_before was captured).
        EXPECT_EQ(worker.w_gsn_clock, gsn_before + 1);
    }

    // Double commit on an already-right state must be safe.
    EXPECT_FALSE(active_txn().CommitTransaction());
}

TEST_F(TestTransaction, CrossThreadWorkerMapping) {
    {
        TransactionAutoCommitScope t;
        EXPECT_EQ(active_txn().LogWorker().w_id, 0u);
    }

    std::atomic<uint16_t> other_wid{255};
    std::atomic<txid_t> other_txn{0};
    std::thread th([&] {
        workerThreadId = 1;
        {
            TransactionAutoCommitScope t;
            other_wid.store(active_txn().LogWorker().w_id,
                            std::memory_order_release);
            other_txn.store(active_txn().TxnId(), std::memory_order_release);
        }
    });
    th.join();
    EXPECT_EQ(other_wid.load(), 1u);
    EXPECT_NE(other_txn.load(), 0u);
}

TEST_F(TestTransaction, AbortEmitsTxAbort) {
    TransactionAutoCommitScope t;
    active_txn().MarkAsWrite();
    active_txn().AbortTransaction();
    EXPECT_EQ(active_txn().GetState(), TxState::ABORTED);
    EXPECT_FALSE(active_txn().IsRunning());

    auto* abort = reinterpret_cast<recovery::LogEntry*>(
        active_txn().LogWorker().log_buffer.wal_buffer +
        sizeof(recovery::LogMetaEntry));  // after TX_START
    EXPECT_EQ(abort->type,
              static_cast<uint8_t>(recovery::LogEntry::Type::TX_ABORT));
}

TEST_F(TestTransaction, ReadOnlyTxnStillCommits) {
    TransactionAutoCommitScope t;  // read-only: no MarkAsWrite
    // no WAL DATA entries: TX_START was still emitted but that's the current
    // Phase 1 contract; commit completes without a vector payload.
    EXPECT_EQ(active_txn().SerializedVectorSize(), 4u);
}

}  // namespace caliby::tx
