#include "recovery/checkpoint.hpp"

#include "calico.hpp"
#include "logging.hpp"
#include "recovery/log_manager.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace caliby {
namespace recovery {

Checkpointer::Checkpointer(LogManager* log_manager, uint32_t num_shards)
    : log_manager_(log_manager), num_shards(num_shards) {
    shard_watermark_.reserve(num_shards);
    for (uint32_t i = 0; i < num_shards; ++i) {
        shard_watermark_.emplace_back(std::make_unique<std::atomic<gsn_t>>(0));
    }
}

Checkpointer::~Checkpointer() {
    Stop();
}

void Checkpointer::Start() {
    if (th_.joinable()) return;
    stop_.store(false, std::memory_order_release);
    th_ = std::thread([this]() { RunLoop(); });
}

void Checkpointer::Stop() {
    stop_.store(true, std::memory_order_release);
    if (th_.joinable()) th_.join();
}

void Checkpointer::RunLoop() {
    while (!stop_.load(std::memory_order_acquire)) {
        if (enabled) {
            TryCheckpoint();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
}

void Checkpointer::TryCheckpoint() {
    if (!log_manager_) return;
    // Compute the durable frontier.
    gsn_t durable = log_manager_->global_min_gsn_flushed.load(std::memory_order_acquire);
    if (durable == 0) return;

    // Round-robin shard.
    uint32_t shard = static_cast<uint32_t>(increment_++ % num_shards);

    // The "flush one shard" path normally would call into BufferManager. In
    // Caliby we use a simpler invariant: ask the BufferManager to flush any
    // page whose GSN <= durable and whose PID maps to `shard`. The Buffer
    // manager keeps a per-shard dirty list; the actual implementation lives
    // in src/calico.cpp.
    //
    // For now we use the catch-all BufferManager::flushAll() if the hook is
    // not set; this is slower but always correct. The shard-flush hook is
    // added in Phase 2.
    if (bm_ptr != nullptr) {
        // Determine a safe subset: all dirty pages whose p_gsn <= durable.
        // We piggyback on flushAll() which already only writes dirty pages.
        // (Phase 2 will add a shard-aware variant.)
        try {
            bm_ptr->flushAll();
        } catch (...) {
            // Don't let flush errors kill the checkpointer thread.
        }
    }

    // Recycle WAL blocks: any CLOSED block with block_gsn <= durable can be
    // marked DONE.
    log_manager_->WALBlocks().MarkDoneThroughGsn(durable);

    // Advance per-shard watermark.
    auto prev = shard_watermark_[shard]->load(std::memory_order_relaxed);
    if (durable > prev) shard_watermark_[shard]->store(durable, std::memory_order_release);

    if (caliby::is_log_enabled(caliby::LogLevel::DEBUG)) {
        CALIBY_LOG_DEBUG("Checkpointer", "shard=", shard, " durable=", durable,
                         " blocks_avail=", log_manager_->WALBlocks().BlockCycleCnt());
    }
}

void Checkpointer::Bootstrap() {
    // After recovery, every shard watermark is initialized to 0 - the
    // checkpointer will start fresh in TryCheckpoint().
    for (auto& a : shard_watermark_) a->store(0, std::memory_order_relaxed);
    increment_ = 0;
}

}  // namespace recovery
}  // namespace caliby
