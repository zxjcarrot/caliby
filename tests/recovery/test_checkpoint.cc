// Cloned from LeanStore/test/recovery/test_checkpoint.cc
// Tests that the checkpointer only recycles CLOSED blocks whose GSN is at
// most the durable frontier.
#include "recovery/checkpoint.hpp"
#include "recovery/log_manager.hpp"
#include "recovery/wal_block_manager.hpp"

#include <gtest/gtest.h>

namespace caliby::recovery {

class TestCheckpointer : public ::testing::Test {
protected:
    std::atomic<bool> running_{true};
    std::unique_ptr<LogManager> log_;
    std::unique_ptr<Checkpointer> cp_;

    void SetUp() override {
        std::string p = "/tmp/caliby_test_checkpoint.wal";
        std::remove(p.c_str());
        log_ = std::make_unique<LogManager>(running_, p);
        log_->wal_block_size = 4ULL * 1024 * 1024;
        // Use enough blocks so OpenNext() can satisfy 5 successive calls.
        // Each call returns the next DONE block (back-pressure when none).
        // We pre-mark blocks 0..3 as DONE (via OpenNext + store) so the 5th
        // call lands on a fresh, properly-stamped block.
        log_->wal_size_mb    = 32;  // 32MB / 4MB = 8 blocks
        log_->buffer_size_mb = 1;
        log_->StartWorkers();
        log_->WriteMasterRecord();
        cp_ = std::make_unique<Checkpointer>(log_.get(), 4);
    }
    void TearDown() override {
        cp_.reset();
        log_.reset();
        running_.store(false);
    }
};

TEST_F(TestCheckpointer, RespectsDurabilityBarrier) {
    constexpr gsn_t DURABLE_GSN = 1002;
    auto& mgr = log_->WALBlocks();

    auto& open_blk     = mgr.OpenNext();
    open_blk.block_gsn = 1000;
    auto& covered_1     = mgr.OpenNext();
    covered_1.block_gsn = 1001;
    covered_1.state.store(static_cast<uint8_t>(WALBlock::State::CLOSED), std::memory_order_release);
    auto& covered_2     = mgr.OpenNext();
    covered_2.block_gsn = DURABLE_GSN;
    covered_2.state.store(static_cast<uint8_t>(WALBlock::State::CLOSED), std::memory_order_release);
    auto& barrier_blk     = mgr.OpenNext();
    barrier_blk.block_gsn = DURABLE_GSN + 1;
    barrier_blk.state.store(static_cast<uint8_t>(WALBlock::State::CLOSED), std::memory_order_release);
    auto& done_blk = mgr.OpenNext();
    done_blk.state.store(static_cast<uint8_t>(WALBlock::State::DONE), std::memory_order_release);

    LogManager::global_min_gsn_flushed.store(DURABLE_GSN, std::memory_order_release);

    auto cnt_before = mgr.BlockCycleCnt();
    mgr.MarkDoneThroughGsn(DURABLE_GSN);
    auto cnt_after = mgr.BlockCycleCnt();
    EXPECT_EQ(cnt_after - cnt_before, 2u);

    EXPECT_EQ(mgr.StateOf(0), WALBlock::State::OPEN);
    EXPECT_EQ(mgr.StateOf(1), WALBlock::State::DONE);
    EXPECT_EQ(mgr.StateOf(2), WALBlock::State::DONE);
    EXPECT_EQ(mgr.StateOf(3), WALBlock::State::CLOSED);
    EXPECT_EQ(mgr.StateOf(4), WALBlock::State::DONE);
}

}  // namespace caliby::recovery
