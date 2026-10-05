// Tests for the per-worker circular LogBuffer: dirty size, wrap-around,
// autonomous flush behavior.
#include "recovery/log_buffer.hpp"
#include "recovery/log_manager.hpp"
#include "recovery/log_worker.hpp"

#include <gtest/gtest.h>

namespace caliby::recovery {

class TestLogBuffer : public ::testing::Test {
protected:
    std::atomic<bool> running_{true};
    std::unique_ptr<LogManager> log_;
    LogWorker* worker_ = nullptr;

    void SetUp() override {
        std::string p = "/tmp/caliby_test_logbuffer.wal";
        std::remove(p.c_str());
        log_ = std::make_unique<LogManager>(running_, p);
        log_->wal_block_size = 4ULL * 1024 * 1024;
        log_->wal_size_mb    = 8;
        log_->buffer_size_mb = 1;
        log_->StartWorkers();
        log_->WriteMasterRecord();
        worker_ = &log_->LocalLogWorker();
    }
    void TearDown() override {
        log_.reset();
        running_.store(false);
    }
};

TEST_F(TestLogBuffer, TotalFreeSpaceEqualsBufferSize) {
    EXPECT_EQ(worker_->log_buffer.TotalFreeSpace(),
              worker_->log_buffer.buffer_size);
}

TEST_F(TestLogBuffer, SubmitEntryReducesFreeSpace) {
    auto& e = worker_->ReserveLogMetaEntry();
    e.type = static_cast<uint8_t>(LogEntry::Type::TX_START);
    worker_->SubmitActiveLogEntry();
    EXPECT_EQ(worker_->log_buffer.wal_cursor, sizeof(LogMetaEntry));
    EXPECT_EQ(worker_->log_buffer.TotalFreeSpace(),
              worker_->log_buffer.buffer_size - sizeof(LogMetaEntry));
}

TEST_F(TestLogBuffer, DataEntryRoundTrip) {
    auto& dt = worker_->ReserveDataLog(/*payload=*/64, /*pid=*/7);
    dt.rc.w_id = 0;
    dt.payload();  // valid
    worker_->SubmitActiveLogEntry();

    auto* e = reinterpret_cast<LogEntry*>(worker_->log_buffer.wal_buffer);
    EXPECT_EQ(e->type, static_cast<uint8_t>(LogEntry::Type::DATA_ENTRY));
    EXPECT_EQ(e->size, sizeof(DataEntry) + 64);
    auto* raw = reinterpret_cast<DataEntry*>(e);
    EXPECT_EQ(raw->pid, 7u);
    EXPECT_EQ(raw->rc.w_id, 0u);
}

}  // namespace caliby::recovery
