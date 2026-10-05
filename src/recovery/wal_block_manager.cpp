#include "recovery/wal_block_manager.hpp"

#include <algorithm>
#include <sched.h>

namespace caliby {
namespace recovery {

WALBlockManager::WALBlockManager(uint32_t num_blocks, uint64_t wal_block_size, uint64_t wal_region_size)
    : num_blocks_(num_blocks), wal_block_size_(wal_block_size), wal_region_size_(wal_region_size) {
    blocks_.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        blocks_.emplace_back(std::make_unique<WALBlock>(static_cast<uint64_t>(i) * wal_block_size, wal_block_size));
    }
}

uint64_t WALBlockManager::BlockCycleCnt() {
    uint64_t n = 0;
    for (auto& bp : blocks_) {
        if (bp->state.load(std::memory_order_acquire) == static_cast<uint8_t>(WALBlock::State::DONE)) {
            ++n;
        }
    }
    return n;
}

WALBlock& WALBlockManager::OpenNext() {
    // Round-robin scan starting from the first DONE block we find. We don't
    // need a static cursor; we just keep trying until something works.
    // Back-pressure: if no block is currently DONE, this loop spins. The
    // checkpointer is the only thing that creates DONE blocks, so the loop
    // terminates as soon as the checkpointer makes progress.
    for (;;) {
        for (uint32_t i = 0; i < num_blocks_; ++i) {
            WALBlock& b = *blocks_[i];
            uint8_t expected = static_cast<uint8_t>(WALBlock::State::DONE);
            if (b.state.compare_exchange_strong(expected, static_cast<uint8_t>(WALBlock::State::OPEN),
                                               std::memory_order_acq_rel, std::memory_order_acquire)) {
                // Reset block state for the new occupant.
                b.block_gsn = 0;
                b.block_next.store(0, std::memory_order_release);
                return b;
            }
        }
        sched_yield();
    }
}

void WALBlockManager::MarkDoneThroughGsn(gsn_t gsn) {
    for (uint32_t i = 0; i < num_blocks_; ++i) {
        WALBlock& b = *blocks_[i];
        if (b.state.load(std::memory_order_acquire) != static_cast<uint8_t>(WALBlock::State::CLOSED)) {
            continue;
        }
        if (b.block_gsn <= gsn) {
            // Atomically transition CLOSED -> DONE. Another thread may have
            // already done this; that's fine.
            uint8_t expected = static_cast<uint8_t>(WALBlock::State::CLOSED);
            b.state.compare_exchange_strong(expected, static_cast<uint8_t>(WALBlock::State::DONE),
                                            std::memory_order_acq_rel, std::memory_order_acquire);
        }
    }
}

void WALBlockManager::RecoverBlock(uint32_t idx, gsn_t gsn) {
    if (idx >= num_blocks_) return;
    WALBlock& b = *blocks_[idx];
    b.block_gsn = gsn;
    b.block_next.store(0, std::memory_order_release);
    b.state.store(static_cast<uint8_t>(WALBlock::State::CLOSED), std::memory_order_release);
}

}  // namespace recovery
}  // namespace caliby
