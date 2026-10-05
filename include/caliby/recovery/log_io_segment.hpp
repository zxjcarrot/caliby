#pragma once

#include "log_entry.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace caliby {
namespace recovery {

// Owns a temporary buffer used for serializing / deserializing a chunk of WAL
// records (a single LogIOSegment = up to one block's worth of data).
struct LogIOSegment {
    uint8_t* buffer = nullptr;   // current write segment (owned by the worker)

    // Heap-allocated buffer wrapper used to carry a deserialized chunk
    // through recovery. May "steal" memory from a raw pointer.
    struct Buffer {
        uint8_t* data       = nullptr;
        uint64_t size       = 0;
        bool     stolen     = false;  // true => don't free

        Buffer() = default;
        explicit Buffer(uint64_t sz) : data(new uint8_t[sz]()), size(sz), stolen(false) {}
        Buffer(uint8_t* stolen_data, uint64_t sz) : data(stolen_data), size(sz), stolen(true) {}
        Buffer(Buffer&& o) noexcept : data(o.data), size(o.size), stolen(o.stolen) {
            o.data = nullptr;
            o.size = 0;
        }
        Buffer& operator=(Buffer&& o) noexcept {
            if (this != &o) {
                Deallocate();
                data = o.data;
                size = o.size;
                stolen = o.stolen;
                o.data = nullptr;
                o.size = 0;
            }
            return *this;
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        ~Buffer() { Deallocate(); }

        void Deallocate() {
            if (data && !stolen) {
                delete[] data;
            }
            data = nullptr;
            size = 0;
            stolen = false;
        }

        // Append a serialized log entry to the buffer.
        void InsertNewLog(const uint8_t* log, uint64_t log_size) {
            // Note: this is only used in tests; for production we go through
            // the LogBuffer/LogBackend path.
            if (!data) {
                data = new uint8_t[log_size]();
                size = log_size;
            } else {
                auto* new_data = new uint8_t[size + log_size]();
                std::memcpy(new_data, data, size);
                std::memcpy(new_data + size, log, log_size);
                delete[] data;
                data = new_data;
                size = size + log_size;
            }
        }

        bool IsEmpty() const { return size == 0; }

        // Walk every log entry. `fn` returns false to stop iteration.
        uint64_t Iterate(const std::function<bool(const LogEntry*)>& fn) const;

        // Slice [offset, offset+len)
        std::pair<const uint8_t*, uint64_t> Slice(uint64_t offset, uint64_t len) const {
            return {data + offset, len};
        }
    };

    // Tournament-tree merge generator over multiple per-worker buffers.
    // Produces DataEntry records in ascending GSN order, deduplicating
    // entries that have the same (pid, gsn).
    struct LazyGenerator {
        struct Offset {
            Buffer* buf;
            uint64_t off;       // next byte to read
            Offset(Buffer* b, uint64_t o = 0) : buf(b), off(o) {}
        };
        struct Compare {
            bool operator()(const Offset& a, const Offset& b) const {
                if (!a.buf) return true;
                if (!b.buf) return false;
                // Read current entry's gsn
                gsn_t ga = 0, gb = 0;
                if (a.off + sizeof(DataEntry) <= a.buf->size) {
                    auto* e = reinterpret_cast<const DataEntry*>(a.buf->data + a.off);
                    ga = e->rc.gsn;
                }
                if (b.off + sizeof(DataEntry) <= b.buf->size) {
                    auto* e = reinterpret_cast<const DataEntry*>(b.buf->data + b.off);
                    gb = e->rc.gsn;
                }
                return ga > gb;  // min-heap
            }
        };
        std::priority_queue<Offset, std::vector<Offset>, Compare> tournament;

        void ReceiveInput(Buffer* b) {
            if (b && !b->IsEmpty()) {
                tournament.push(Offset{b, 0});
            }
        }

        // Walk merged entries in ascending GSN. `fn` returns false to stop.
        uint64_t Iterate(const std::function<bool(const DataEntry*)>& fn);
    };

    LogIOSegment() = default;
    ~LogIOSegment() = default;

    // Serializes a chunk by calling `fill_fn(dst, size_in_out)` to write log
    // records into `dst`. After the records, appends a WRITE_METADATA entry
    // containing the chunk signature. Then calls `write_fn(dst, size)` to
    // hand the buffer to the backend.
    void Serialize(uint64_t reserved_size,
                   const std::function<void(uint8_t*, uint64_t&)>& fill_fn,
                   const std::function<void(uint8_t*, uint64_t)>& write_fn);

    // Reads from `buffer` (which is read-only, points at a chunk in the WAL
    // file). Walks entries, calls `analyze_fn(buf, len)` for each non-meta
    // entry. Returns the offset (within `buffer`) of the WRITE_METADATA
    // record that was found.
    static uint64_t Deserialize(uint8_t* buffer, uint64_t end_offset, uint32_t required_signature,
                                const std::function<void(uint8_t*, uint64_t)>& analyze_fn);
};

}  // namespace recovery
}  // namespace caliby
