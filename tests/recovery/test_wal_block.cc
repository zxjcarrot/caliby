// Cloned from LeanStore/test/recovery/test_wal_block.cc
// Tests the logical ring of WAL blocks: open, close, recycle, back-pressure.

#include "recovery/wal_block_manager.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace caliby::recovery {

static constexpr uint64_t kTestWalBlockSize   = 4ULL * 1024 * 1024;       // 4 MB
static constexpr uint64_t kTestWalRegionSize  = kTestWalBlockSize * 20;   // 80 MB

TEST(TestWALBlock, RingWrapAround) {
    auto num_blocks = 16U;
    WALBlockManager mgr(num_blocks, kTestWalBlockSize, kTestWalRegionSize);
    EXPECT_EQ(mgr.Capacity(), num_blocks);

    constexpr auto TOTAL = 40UL;
    for (uint64_t i = 0UL; i < TOTAL; i++) {
        auto& blk     = mgr.OpenNext();
        blk.block_gsn = i + 7;
        blk.state.store(static_cast<uint8_t>(WALBlock::State::DONE));
    }
}

TEST(TestWALBlock, StampingAndReuseState) {
    WALBlockManager mgr(4, kTestWalBlockSize, kTestWalRegionSize);

    auto& b0     = mgr.OpenNext();
    b0.block_gsn = 10;
    b0.state.store(static_cast<uint8_t>(WALBlock::State::DONE), std::memory_order_release);

    for (uint64_t i = 0; i < 3; i++) {
        auto& blk = mgr.OpenNext();
        blk.state.store(static_cast<uint8_t>(WALBlock::State::DONE), std::memory_order_release);
    }

    auto& reclaimed = mgr.OpenNext();
    EXPECT_EQ(reclaimed.state.load(std::memory_order_acquire),
              static_cast<uint8_t>(WALBlock::State::OPEN));
    EXPECT_EQ(reclaimed.block_gsn, 0u);
}

TEST(TestWALBlock, BackpressureWaitsForDone) {
    WALBlockManager mgr(4, kTestWalBlockSize, kTestWalRegionSize);

    for (uint64_t i = 0; i < 4; i++) { (void)mgr.OpenNext(); }

    std::atomic<bool> opened{false};
    std::thread writer([&]() {
        mgr.OpenNext();
        opened.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(opened.load());

    mgr.Block(0).state.store(static_cast<uint8_t>(WALBlock::State::DONE), std::memory_order_release);
    writer.join();
    EXPECT_TRUE(opened.load());
}

TEST(TestWALBlock, BackpressureProceedsWhenDone) {
    WALBlockManager mgr(4, kTestWalBlockSize, kTestWalRegionSize);

    for (uint64_t i = 0; i < 4; i++) {
        auto& blk     = mgr.OpenNext();
        blk.block_gsn = 100 + i;
        blk.state.store(static_cast<uint8_t>(WALBlock::State::DONE), std::memory_order_release);
    }

    auto& b4 = mgr.OpenNext();
    EXPECT_EQ(b4.state.load(std::memory_order_acquire),
              static_cast<uint8_t>(WALBlock::State::OPEN));
    EXPECT_EQ(b4.block_gsn, 0u);
}

TEST(TestWALBlock, MarkDoneThroughGsnRecyclesCoveredPrefix) {
    WALBlockManager mgr(4, kTestWalBlockSize, kTestWalRegionSize);

    for (uint64_t i = 0; i < 3; i++) {
        auto& blk     = mgr.OpenNext();
        blk.block_gsn = 100 + i;
        blk.state.store(static_cast<uint8_t>(WALBlock::State::CLOSED), std::memory_order_release);
    }

    EXPECT_EQ(mgr.BlockCycleCnt(), 1u);

    mgr.MarkDoneThroughGsn(101);
    EXPECT_EQ(mgr.BlockCycleCnt(), 3u);
    EXPECT_EQ(mgr.StateOf(0), WALBlock::State::DONE);
    EXPECT_EQ(mgr.StateOf(1), WALBlock::State::DONE);
    EXPECT_EQ(mgr.StateOf(2), WALBlock::State::CLOSED);
    EXPECT_EQ(mgr.StateOf(3), WALBlock::State::DONE);

    mgr.MarkDoneThroughGsn(102);
    EXPECT_EQ(mgr.BlockCycleCnt(), 4u);
    EXPECT_EQ(mgr.StateOf(2), WALBlock::State::DONE);
}

}  // namespace caliby::recovery
