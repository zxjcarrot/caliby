#include "recovery/recovery_backend.hpp"

#include "logging.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

namespace caliby {
namespace recovery {

void RecoveryBackend::Connect() {
    if (!log_manager_) return;
    wal_fd_ = ::open(log_manager_->wal_path().c_str(), O_RDONLY | O_CLOEXEC);
    if (wal_fd_ < 0) {
        CALIBY_LOG_WARN("RecoveryBackend", "open(", log_manager_->wal_path(), ") failed");
    }
}

bool RecoveryBackend::RetrieveMasterRecord(MasterRecordEntry& out) {
    if (wal_fd_ < 0) return false;
    uint8_t buf[4096] = {0};
    ssize_t n = ::pread(wal_fd_, buf, sizeof(buf), 0);
    if (n < static_cast<ssize_t>(sizeof(MasterRecordEntry))) return false;
    std::memcpy(&out, buf, sizeof(out));
    if (out.type != static_cast<uint8_t>(LogEntry::Type::MASTER_RECORD)) return false;
    if (out.size != sizeof(MasterRecordEntry)) return false;

    signature_             = out.signature;
    prev_worker_count_     = out.num_workers;
    prev_wal_size_mb_      = out.wal_size_mb;
    prev_wal_block_size_mb_ = out.wal_block_size_mb;
    wal_block_size_        = static_cast<uint64_t>(out.wal_block_size_mb) * 1024 * 1024;
    return true;
}

uint64_t RecoveryBackend::ReadLogBlocks(
    const std::function<void(uint8_t* buf, uint64_t len, uint32_t signature, uint64_t offset)>& materialize_fn) {
    if (wal_fd_ < 0 || wal_block_size_ == 0) return 0;
    uint64_t region_start = 4 * 1024;
    uint64_t total = static_cast<uint64_t>(prev_wal_size_mb_) * 1024 * 1024;
    uint64_t num_blocks = total / wal_block_size_;
    auto* tmp = new uint8_t[wal_block_size_];
    uint64_t processed = 0;
    for (uint64_t i = 0; i < num_blocks; ++i) {
        ssize_t n = ::pread(wal_fd_, tmp, wal_block_size_, region_start + i * wal_block_size_);
        if (n <= 0) continue;
        // Inspect the last LogMetaEntry in the block to find signature.
        uint32_t sig = 0;
        if (n >= static_cast<ssize_t>(sizeof(LogMetaEntry))) {
            auto* meta = reinterpret_cast<LogMetaEntry*>(tmp + n - sizeof(LogMetaEntry));
            if (meta->type == static_cast<uint8_t>(LogEntry::Type::WRITE_METADATA)) {
                sig = meta->rc.signature;
            }
        }
        materialize_fn(tmp, static_cast<uint64_t>(n), sig, region_start + i * wal_block_size_);
        ++processed;
    }
    delete[] tmp;
    return processed;
}

}  // namespace recovery
}  // namespace caliby
