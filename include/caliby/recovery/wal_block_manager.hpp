#pragma once

#include "log_entry.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace caliby {
namespace recovery {

// One block in the circular WAL region. We use a logical ring (not a file
// tail-truncate ring): a block is reused by overwriting in place. The block
// manager tracks which blocks are free.
//
// State machine:
//   DONE  -> DONE   : recyclable
//   DONE  -> OPEN   : reserved by a writer (atomic CAS)
//   OPEN  -> CLOSED : writer finished, block_gsn is final
//   CLOSED-> DONE   : checkpoint acknowledged - safe to recycle
struct WALBlock {
    enum class State : uint8_t {
        OPEN   = 0,  // currently being written by a LogWorker
        CLOSED = 1,  // writer finished, has final block_gsn stamped
        DONE   = 2,  // checkpointed - safe to overwrite
    };

    const uint64_t block_base;   // lower offset within the WAL region
    uint64_t       block_size;   // bytes
    gsn_t          block_gsn{0}; // max GSN contained; 0 if OPEN
    std::atomic<uint8_t> state{static_cast<uint8_t>(State::DONE)};
    // ascending write frontier (next free byte within the block).
    // We DO NOT wrap inside a block; once we hit block_size we CLOSE it and
    // move on. This keeps writes sequential and predictable.
    std::atomic<uint64_t> block_next{0};

    WALBlock(uint64_t base, uint64_t size)
        : block_base(base), block_size(size) {}

    // Atomics are non-movable; explicitly delete copy/move to surface the
    // issue clearly. The block manager heap-allocates WALBlock to avoid
    // std::vector's requirement for move/copy-constructibility.
    WALBlock(const WALBlock&) = delete;
    WALBlock& operator=(const WALBlock&) = delete;
    WALBlock(WALBlock&&) = delete;
    WALBlock& operator=(WALBlock&&) = delete;
};

// Fixed-capacity logical ring of WAL blocks.
//
// We store std::unique_ptr<WALBlock> internally because WALBlock contains
// std::atomic members which are not move-constructible; the outer
// WALBlockManager is still movable (the unique_ptrs are).
class WALBlockManager {
public:
    WALBlockManager(uint32_t num_blocks, uint64_t wal_block_size, uint64_t wal_region_size);
    ~WALBlockManager() = default;

    uint32_t Capacity() const { return num_blocks_; }
    uint64_t BlockSize() const { return wal_block_size_; }
    uint64_t RegionSize() const { return wal_region_size_; }

    // Number of blocks currently in DONE state.
    uint64_t BlockCycleCnt();

    // Block descriptor by index (used by tests, recovery, etc.)
    WALBlock& Block(uint32_t idx) { return *blocks_[idx]; }
    const WALBlock& Block(uint32_t idx) const { return *blocks_[idx]; }

    // Reserve the next free block. If all blocks are in flight, this spins
    // until one becomes DONE (back-pressure). When we say "spin" we mean a
    // short CAS loop with sched_yield; the checkpointer eventually marks
    // blocks DONE so writers always make progress.
    WALBlock& OpenNext();

    // Recycle every block whose block_gsn <= gsn and which is in CLOSED state.
    void MarkDoneThroughGsn(gsn_t gsn);

    // Mark a previously-recovered block as CLOSED with the given GSN stamp.
    // Used by RecoveryManager during recovery.
    void RecoverBlock(uint32_t idx, gsn_t gsn);

    // For testing/recovery: direct state read.
    WALBlock::State StateOf(uint32_t idx) const {
        return static_cast<WALBlock::State>(blocks_[idx]->state.load(std::memory_order_acquire));
    }

private:
    std::vector<std::unique_ptr<WALBlock>> blocks_;
    const uint32_t        num_blocks_;
    const uint64_t        wal_block_size_;
    const uint64_t        wal_region_size_;
};

}  // namespace recovery
}  // namespace caliby
