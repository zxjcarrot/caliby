#include "recovery/log_manager.hpp"
#include "logging.hpp"
#include "recovery/checkpoint.hpp"
#include "recovery/log_io_segment.hpp"
#include "calico.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace caliby {
namespace recovery {

std::atomic<gsn_t> LogManager::global_min_gsn_flushed{0};
std::atomic<gsn_t> LogManager::global_sync_to_this_gsn{0};

static uint64_t env_u64(const char* name, uint64_t def) {
    if (const char* v = std::getenv(name)) {
        try { return std::stoull(v); } catch (...) {}
    }
    return def;
}

ssize_t pwrite_all(int fd, const void* buf, size_t n_bytes, off_t offset) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
    size_t total = 0;
    while (total < n_bytes) {
        ssize_t n = ::pwrite(fd, p + total, n_bytes - total, offset + total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break;
        total += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(total);
}

int fdatasync_fd(int fd) {
    return ::fdatasync(fd);
}

LogManager::LogManager(std::atomic<bool>& is_running, const std::string& wal_path)
    : is_running_(&is_running), wal_path_(wal_path) {
    SetInstance(this);
    wal_block_size = env_u64("CALIBY_WAL_BLOCK_SIZE_MB", 1) * 1024ull * 1024;
    wal_size_mb    = env_u64("CALIBY_WAL_SIZE_MB", 64);
    buffer_size_mb = env_u64("CALIBY_WAL_BUFFER_SIZE_MB", 4);
    num_workers    = static_cast<uint32_t>(env_u64("CALIBY_NUM_WORKERS", 1));
    if (num_workers == 0) num_workers = 1;
    if (num_workers > MAX_NUMBER_OF_WORKERS) num_workers = MAX_NUMBER_OF_WORKERS;

    if (wal_path_.empty()) {
        wal_path_ = "caliby.wal";
    }
}

void LogManager::StartWorkers() {
    // Open the WAL file and (re)build the block manager now that the caller
    // has had a chance to override tunables.
    OpenWalFile();
    uint32_t num_blocks = static_cast<uint32_t>(wal_size_mb * 1024ull * 1024 / wal_block_size);
    if (num_blocks == 0) num_blocks = 1;
    wal_blocks_ = std::make_unique<WALBlockManager>(num_blocks, wal_block_size,
                                                    static_cast<uint64_t>(num_blocks) * wal_block_size);

    workers_.clear();
    w_state_.clear();
    commit_state_.clear();
    for (uint32_t i = 0; i < num_workers; ++i) {
        workers_.emplace_back(std::make_unique<LogWorker>(i, this, is_running_));
        w_state_.emplace_back();
        commit_state_.emplace_back();
    }
    next_gsn_.store(1, std::memory_order_release);

    StartBackgroundThreads();

    CALIBY_LOG_INFO("LogManager", "Initialized: wal_path=", wal_path_,
                    " wal_size_mb=", wal_size_mb,
                    " block_size_mb=", wal_block_size / (1024 * 1024),
                    " num_blocks=", num_blocks,
                    " num_workers=", num_workers,
                    " buffer_size_mb=", buffer_size_mb);
}

LogManager::~LogManager() {
    StopBackgroundThreads();
    if (wal_fd_ >= 0) {
        ::close(wal_fd_);
        wal_fd_ = -1;
    }
    if (s_instance_ == this) s_instance_ = nullptr;
}

LogManager& LogManager::Instance() {
    assert(s_instance_ != nullptr && "LogManager::SetInstance must be called first");
    return *s_instance_;
}

void LogManager::SetInstance(LogManager* mgr) {
    s_instance_ = mgr;
}

void LogManager::OpenWalFile() {
    int fd = ::open(wal_path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        CALIBY_LOG_ERROR("LogManager", "open(" + wal_path_ + ") failed");
        return;
    }
    wal_fd_ = fd;
    // Ensure file is at least 4KB + wal_size_mb bytes.
    uint64_t total = (4 * 1024) + wal_size_mb * 1024ull * 1024;
    struct stat st {};
    if (::fstat(fd, &st) == 0 && static_cast<uint64_t>(st.st_size) < total) {
        if (::ftruncate(fd, total) != 0) {
            CALIBY_LOG_WARN("LogManager", "ftruncate to ", total, " failed");
        }
    }
    wal_region_start_ = 4 * 1024;
    wal_region_end_   = wal_region_start_ + wal_size_mb * 1024ull * 1024;
}

void LogManager::StartBackgroundThreads() {
    if (checkpointer_) return;  // already running
    if (env_u64("CALIBY_WAL_CHECKPOINT", 1) != 0) {
        checkpointer_ = std::make_unique<Checkpointer>(this);
        checkpointer_->Start();
    }
}

void LogManager::StopBackgroundThreads() {
    if (checkpointer_) {
        checkpointer_->Stop();
        checkpointer_.reset();
    }
}

LogWorker& LogManager::LocalLogWorker() {
    // Map the thread-local Caliby worker id onto a LogWorker slot. Threads
    // beyond num_workers round-robin onto the existing workers (safe: each
    // LogWorker is an independent buffer and matches LeanStore sub-worker
    // sharing).
    uint16_t tid = workerThreadId;
    uint32_t slot = static_cast<uint32_t>(tid) % static_cast<uint32_t>(workers_.size());
    return *workers_[slot];
}

WALBlock* LogManager::OpenNextBlock() {
    auto& blk = wal_blocks_->OpenNext();
    return &blk;
}

gsn_t LogManager::AdvanceFlushWatermark() {
    gsn_t min_gsn = std::numeric_limits<gsn_t>::max();
    for (auto& s : w_state_) {
        min_gsn = std::min(min_gsn, s.last_gsn);
    }
    if (min_gsn == std::numeric_limits<gsn_t>::max()) min_gsn = 0;
    auto cur = global_min_gsn_flushed.load(std::memory_order_acquire);
    while (min_gsn > cur) {
        if (global_min_gsn_flushed.compare_exchange_weak(cur, min_gsn,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
            break;
        }
    }
    return global_min_gsn_flushed.load(std::memory_order_relaxed);
}

void LogManager::WriteMasterRecord() {
    if (wal_fd_ < 0) return;
    MasterRecordEntry rec{};
    rec.type = static_cast<uint32_t>(LogEntry::Type::MASTER_RECORD);
    rec.size = sizeof(MasterRecordEntry);
    rec.chksum = 0;
    rec.rc = RecoveryInfo{};
    rec.num_workers       = num_workers;
    rec.wal_size_mb       = static_cast<uint32_t>(wal_size_mb);
    rec.wal_block_size_mb = static_cast<uint32_t>(wal_block_size / (1024 * 1024));
    rec.signature         = 0xCAFEBABE;
    rec.max_pid           = 0;
    rec.next_gsn          = next_gsn_.load(std::memory_order_relaxed);
    rec.ComputeChksum();
    uint8_t buf[4096] = {0};
    std::memcpy(buf, &rec, sizeof(rec));
    pwrite_all(wal_fd_, buf, sizeof(buf), 0);
    fdatasync_fd(wal_fd_);
}

bool LogManager::ReadMasterRecord(MasterRecordEntry& out) {
    if (wal_fd_ < 0) return false;
    uint8_t buf[4096] = {0};
    ssize_t n = ::pread(wal_fd_, buf, sizeof(buf), 0);
    if (n < static_cast<ssize_t>(sizeof(MasterRecordEntry))) return false;
    std::memcpy(&out, buf, sizeof(out));
    if (out.type != static_cast<uint32_t>(LogEntry::Type::MASTER_RECORD)) return false;
    if (out.size != sizeof(MasterRecordEntry)) return false;
    return true;
}

}  // namespace recovery
}  // namespace caliby
