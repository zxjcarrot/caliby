#pragma once

#include "log_entry.hpp"
#include "log_manager.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace caliby {
namespace recovery {

// Provides I/O during recovery. Reads the previous session's master record,
// then walks the WAL blocks and materializes their content for analysis.
class RecoveryBackend {
public:
    explicit RecoveryBackend(LogManager* log_manager) : log_manager_(log_manager) {}

    // Open / refresh the WAL file descriptor (separate from the live one).
    void Connect();

    // Read the master record at offset 0 of the WAL file.
    bool RetrieveMasterRecord(MasterRecordEntry& out);

    // Walk every block in the WAL region and call `materialize_fn` for each
    // one. The callback receives the block's bytes, its length, and the
    // signature stored in the WRITE_METADATA at the end (or 0 if absent).
    //
    // Returns the number of blocks processed.
    uint64_t ReadLogBlocks(
        const std::function<void(uint8_t* buf, uint64_t len, uint32_t signature, uint64_t offset)>& materialize_fn);

    // The previous session's worker count / wal size, etc.
    uint32_t PrevWorkerCount() const { return prev_worker_count_; }
    uint32_t PrevWalSizeMB() const { return prev_wal_size_mb_; }
    uint32_t PrevWalBlockSizeMB() const { return prev_wal_block_size_mb_; }
    uint32_t Signature() const { return signature_; }
    uint64_t WalBlockSize() const { return wal_block_size_; }

private:
    LogManager* log_manager_ = nullptr;
    int         wal_fd_ = -1;
    uint32_t    signature_ = 0;
    uint64_t    wal_block_size_ = 0;
    uint32_t    prev_worker_count_ = 0;
    uint32_t    prev_wal_size_mb_ = 0;
    uint32_t    prev_wal_block_size_mb_ = 0;
};

}  // namespace recovery
}  // namespace caliby
