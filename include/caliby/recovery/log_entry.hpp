#pragma once

#include <cstdint>
#include <unistd.h>

#include <cstring>

namespace caliby {
namespace recovery {

// GSN: Global Sequence Number - monotonic ordering of log records across all workers
using gsn_t = uint64_t;
// Transaction ID
using txid_t = uint64_t;
// Worker ID (logical index 0..MAX_WORKERS-1)
using wid_t = uint16_t;
// Page ID (matches Caliby PID = u64)
using pageid_t = uint64_t;
// Logical / local offset within a worker's circular WAL buffer
using wal_offset_t = uint64_t;
static constexpr wid_t MAX_NUMBER_OF_WORKERS = 128;
static constexpr txid_t INVALID_TXN_ID = 0;
static constexpr gsn_t INVALID_GSN = 0;
static constexpr pageid_t INVALID_PID = UINT64_MAX;
// Short aliases for code-portability.
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// 8-byte fixed header for every WAL record.
// Layout (little-endian):
//   bits 0..3  : type    (LogEntry::Type)
//   bits 4..31 : size    (total record size, 28 bits - up to 256MB per record)
//   bits 32..63: chksum  (xor-based integrity check)
struct LogEntry {
    enum class Type : uint8_t {
        MASTER_RECORD   = 0,
        WRITE_METADATA  = 1,
        CARRIAGE_RETURN = 2,
        TX_START        = 3,
        TX_COMMIT       = 4,
        TX_ABORT        = 5,
        DATA_ENTRY      = 6,
        FREE_EXTENT     = 9,
        REUSE_EXTENT    = 10,
        ALLOC_NEW_PAGE  = 12,
    };

    // Packed into 4 bytes: 4-bit type (low bits) + 28-bit size (high bits).
    // Bitfield declaration order is low-to-high, so `type` occupies bits
    // 0..3 and `size` occupies bits 4..31.
    // Stored as a single u32 so we can write/read it atomically.
    union {
        struct {
            uint32_t type : 4;
            uint32_t size : 28;
        };
        uint32_t type_and_size = 0;
    };
    uint32_t chksum = 0;

    static constexpr uint32_t CHKSUM_PLACEHOLDER = 99;

    void ComputeChksum() {
        chksum = CHKSUM_PLACEHOLDER;
        // simple xor over preceding bytes (which is what LeanStore does)
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(this);
        uint32_t h = CHKSUM_PLACEHOLDER;
        // XOR over the entry itself, including type/size but not chksum
        size_t n = sizeof(LogEntry) - sizeof(uint32_t);
        for (size_t i = 0; i < n; ++i) {
            h = (h * 31) + bytes[i];
        }
        chksum = h;
    }

    void ValidateChksum() const {
        const_cast<LogEntry*>(this)->ComputeChksum();
        // Caller is expected to assert after this.
    }
};
static_assert(sizeof(LogEntry) == 8, "LogEntry must be 8 bytes");

// Must immediately follow LogEntry in every record that represents a logged
// change (DATA_ENTRY, TX_*, FREE_EXTENT, REUSE_EXTENT, MASTER_RECORD).
//
// For WRITE_METADATA the same layout is reused but the payload means
// (chunk_size, signature) instead of (txn, padding).
struct RecoveryInfo {
    wid_t    w_id   = 0;  // logical worker that produced the record
    gsn_t    gsn    = 0;  // global sequence number (0 for WRITE_METADATA)
    union {
        txid_t txn;       // user transaction id (TX_START/COMMIT/ABORT/DATA/FREE/REUSE)
        struct {
            uint32_t chunk_size;
            uint32_t signature;
        };
    };
};
static_assert(sizeof(RecoveryInfo) == 24, "RecoveryInfo must be 24 bytes");

// Log meta entry: TX_START, TX_ABORT, CARRIAGE_RETURN, WRITE_METADATA.
struct LogMetaEntry : LogEntry {
    RecoveryInfo rc;
};
static_assert(sizeof(LogMetaEntry) == 8 + 24, "LogMetaEntry size");

// Data entry: page modification record.
// After this header follows WAL-type-specific payload (key, value, etc.).
struct DataEntry : LogEntry {
    RecoveryInfo rc;
    pageid_t pid       = 0;  // page that was modified
    gsn_t     prev_gsn = 0;  // previous p_gsn of that page (recovery chain)

    // Variable-length payload follows the fixed DataEntry header. Callers
    // that wrote ReserveDataLog(payload_size, pid) get back a DataEntry whose
    // total size on disk is sizeof(DataEntry) + payload_size.
    uint8_t* payload() { return reinterpret_cast<uint8_t*>(this) + sizeof(DataEntry); }
    const uint8_t* payload() const { return reinterpret_cast<const uint8_t*>(this) + sizeof(DataEntry); }
};
static_assert(sizeof(DataEntry) == 48, "DataEntry must be 48 bytes");

// Written at offset 0 of the WAL file at startup. It carries the previous
// session's config so recovery can know how to read the file layout.
struct MasterRecordEntry : LogEntry {
    RecoveryInfo rc;          // w_id=0, gsn=0, txn=0
    uint32_t num_workers      = 0;
    uint32_t wal_size_mb      = 0;
    uint32_t wal_block_size_mb = 0;
    uint32_t signature        = 0;
    uint32_t max_pid          = 0;     // largest PID ever allocated (recovery hint)
    uint64_t next_gsn         = 0;     // next GSN to issue
    uint8_t  reserved[24]     = {0};
};
static_assert(sizeof(MasterRecordEntry) <= 4096, "MasterRecordEntry must fit in 4KB");

// TX_COMMIT log record. Carries a serialized GSN vector used for
// transaction-dependency tracking during recovery analysis.
struct TxnCommitEntry : LogEntry {
    RecoveryInfo rc;
    uint32_t     vector_size = 0;  // number of bytes following the entry
    // Followed by vector_size bytes of payload: serialized (wid_t, gsn_t) pairs.
    uint8_t* payload() { return reinterpret_cast<uint8_t*>(this) + sizeof(TxnCommitEntry); }
    const uint8_t* payload() const { return reinterpret_cast<const uint8_t*>(this) + sizeof(TxnCommitEntry); }
};

// FREE_EXTENT / REUSE_EXTENT are mostly the same on disk.
struct FreeExtentEntry : LogEntry {
    RecoveryInfo rc;
    pageid_t     start_pid = 0;
    pageid_t     lp_size   = 0;  // number of pages in the extent
};

// ALLOC_NEW_PAGE is special: no RecoveryInfo (it is order-independent, idempotent).
struct AllocNewPageEntry : LogEntry {
    pageid_t pid          = 0;  // always METADATA_PAGE_ID (0) in the current Caliby layout
    pageid_t new_page_id  = 0;
};

}  // namespace recovery
}  // namespace caliby
