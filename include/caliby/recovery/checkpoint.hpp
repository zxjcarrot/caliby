#pragma once

#include "log_entry.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace caliby {
namespace recovery {

class LogManager;
class BufferManager;  // Caliby's buffer manager

// Shard-based fuzzy checkpointer. Each round flushes ~1/S of the dirty
// pages in the buffer pool whose GSN is at most the durable frontier, then
// marks the corresponding WAL blocks as DONE.
class Checkpointer {
public:
    static constexpr uint32_t kDefaultShards = 32;

    explicit Checkpointer(LogManager* log_manager, uint32_t num_shards = kDefaultShards);
    ~Checkpointer();

    // Start the background thread.
    void Start();
    // Stop the background thread (joins).
    void Stop();

    // Trigger one round of checkpointing. Public so tests / Python bindings
    // can drive it deterministically.
    void TryCheckpoint();

    // After recovery: re-stamp per-shard watermarks from on-disk page GSNs.
    void Bootstrap();

    // Configurable knobs.
    uint32_t num_shards = kDefaultShards;
    uint32_t sleep_us   = 1000;     // 1ms between rounds
    bool     enabled    = true;

private:
    void RunLoop();

    LogManager* log_manager_ = nullptr;
    std::atomic<bool> stop_{false};
    std::thread th_;

    // Per-shard watermark: the maximum page GSN we have checkpointed for
    // pages in this shard. Pages with p_gsn <= watermark are safe to recycle.
    //
    // Stored as unique_ptr<atomic> per shard because std::atomic is not
    // move/copy-constructible and thus cannot live directly in a std::vector.
    std::vector<std::unique_ptr<std::atomic<gsn_t>>> shard_watermark_;
    uint64_t increment_ = 0;
};

}  // namespace recovery
}  // namespace caliby
