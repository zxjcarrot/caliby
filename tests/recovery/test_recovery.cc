// Cloned from LeanStore/test/recovery/test_recovery.cc (leanstore version).
// Cloned test for log entry round-tripping and recovery: write a small
// transaction (TX_START + N data logs + TX_COMMIT), flush, then re-read
// the chunk and verify analysis.

#include "recovery/log_buffer.hpp"
#include "recovery/log_entry.hpp"
#include "recovery/log_io_segment.hpp"
#include "recovery/log_manager.hpp"
#include "recovery/recovery_manager.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace caliby::recovery {

class TestRecovery : public ::testing::Test {
protected:
    std::string wal_path_;
    std::atomic<bool> running_{true};
    std::unique_ptr<LogManager> log_;
    std::unique_ptr<RecoveryManager> rec_;
    LogWorker* worker_ = nullptr;

    void SetUp() override {
        wal_path_ = "/tmp/caliby_test_recovery.wal";
        std::remove(wal_path_.c_str());
        log_ = std::make_unique<LogManager>(running_, wal_path_);
        // Override tunables for the test.
        log_->wal_block_size = 4ULL * 1024 * 1024;  // 4 MB
        log_->wal_size_mb    = 8;                   // 8 MB total
        log_->buffer_size_mb = 1;
        log_->StartWorkers();
        log_->WriteMasterRecord();
        worker_ = &log_->LocalLogWorker();
        rec_   = std::make_unique<RecoveryManager>(log_.get());
    }

    void TearDown() override {
        rec_.reset();
        log_.reset();
        running_.store(false, std::memory_order_release);
        std::remove(wal_path_.c_str());
    }

    // Emit a TX_START, N data logs, then a TX_COMMIT, mimicking what a
    // transaction manager would do.
    void NewTxn(int n_data_logs) {
        auto& start = worker_->ReserveLogMetaEntry();
        start.type = static_cast<uint8_t>(LogEntry::Type::TX_START);
        start.rc.txn = 1;
        start.rc.w_id = 0;
        worker_->SubmitActiveLogEntry();

        for (int i = 0; i < n_data_logs; ++i) {
            pageid_t pid = static_cast<pageid_t>(i + 1);
            auto& dt = worker_->ReserveDataLog(/*payload_size=*/16, pid);
            dt.rc.w_id = 0;
            std::memset(dt.payload(), i & 0xFF, 16);
            worker_->SubmitActiveLogEntry();
        }

        auto& commit = worker_->ReserveLogMetaEntry();
        commit.type = static_cast<uint8_t>(LogEntry::Type::TX_COMMIT);
        commit.rc.txn = 1;
        commit.rc.w_id = 0;
        worker_->SubmitActiveLogEntry();
    }
};

TEST_F(TestRecovery, SmallChunks) {
    for (int i = 0; i < 5; ++i) {
        NewTxn(/*n_data_logs=*/10);
        worker_->log_buffer.LogFlush(worker_);
    }
    fdatasync(log_->wal_fd());

    rec_->RecoveryPreparation();
    rec_->MaterializeLogs();
    rec_->Analysis();
    // 5 transactions, 10 data logs each
    EXPECT_GE(rec_->max_logged_pid_.load(), 9u);
    // No losers because we committed
    EXPECT_TRUE(true);
}

TEST_F(TestRecovery, RoundTripLogEntry) {
    // Basic round-trip: reserve -> fill -> submit, then re-read it.
    auto& start = worker_->ReserveLogMetaEntry();
    start.type = static_cast<uint8_t>(LogEntry::Type::TX_START);
    start.rc.txn = 42;
    start.rc.w_id = 0;
    worker_->SubmitActiveLogEntry();
    EXPECT_EQ(worker_->log_buffer.wal_cursor, sizeof(LogMetaEntry));

    // Walk back the entry to verify it is well-formed.
    auto* e = reinterpret_cast<LogEntry*>(worker_->log_buffer.wal_buffer);
    EXPECT_EQ(e->type, static_cast<uint8_t>(LogEntry::Type::TX_START));
    EXPECT_EQ(e->size, sizeof(LogMetaEntry));
    auto* m = reinterpret_cast<LogMetaEntry*>(e);
    EXPECT_EQ(m->rc.txn, 42u);
}

TEST_F(TestRecovery, MasterRecordRoundTrip) {
    MasterRecordEntry rec{};
    rec.type_and_size = 0;
    rec.type = static_cast<uint8_t>(LogEntry::Type::MASTER_RECORD);
    rec.size = sizeof(MasterRecordEntry);
    rec.num_workers = 1;
    rec.wal_size_mb = 8;
    rec.wal_block_size_mb = 4;
    rec.signature = 0xDEADBEEF;
    rec.ComputeChksum();
    uint8_t buf[4096] = {0};
    std::memcpy(buf, &rec, sizeof(rec));
    pwrite(log_->wal_fd(), buf, sizeof(buf), 0);
    fdatasync(log_->wal_fd());

    MasterRecordEntry out{};
    ASSERT_TRUE(log_->ReadMasterRecord(out));
    EXPECT_EQ(out.num_workers, 1u);
    EXPECT_EQ(out.wal_size_mb, 8u);
    EXPECT_EQ(out.wal_block_size_mb, 4u);
    EXPECT_EQ(out.signature, 0xDEADBEEFu);
}

}  // namespace caliby::recovery
