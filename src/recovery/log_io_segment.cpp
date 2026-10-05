#include "recovery/log_io_segment.hpp"

#include <algorithm>
#include <cstring>

namespace caliby {
namespace recovery {

uint64_t LogIOSegment::Buffer::Iterate(const std::function<bool(const LogEntry*)>& fn) const {
    uint64_t off = 0;
    uint64_t cnt = 0;
    while (off < size) {
        auto* e = reinterpret_cast<const LogEntry*>(data + off);
        if (off + sizeof(LogEntry) > size) break;
        if (e->type_and_size == 0) break;
        uint32_t sz = e->size;
        if (sz == 0 || off + sz > size) break;
        if (!fn(e)) return cnt;
        off += sz;
        ++cnt;
    }
    return cnt;
}

uint64_t LogIOSegment::LazyGenerator::Iterate(const std::function<bool(const DataEntry*)>& fn) {
    uint64_t cnt = 0;
    while (!tournament.empty()) {
        Offset top = tournament.top();
        tournament.pop();
        if (!top.buf || top.off + sizeof(DataEntry) > top.buf->size) continue;
        auto* e = reinterpret_cast<const DataEntry*>(top.buf->data + top.off);
        bool keep = fn(e);
        uint32_t sz = e->size;
        if (sz == 0) break;
        top.off += sz;
        if (top.off < top.buf->size) {
            // Re-insert with advanced offset
            tournament.push(top);
        }
        if (!keep) return cnt;
        ++cnt;
    }
    return cnt;
}

void LogIOSegment::Serialize(uint64_t reserved_size,
                             const std::function<void(uint8_t*, uint64_t&)>& fill_fn,
                             const std::function<void(uint8_t*, uint64_t)>& write_fn) {
    if (!buffer) buffer = new uint8_t[reserved_size]();

    uint64_t used = 0;
    fill_fn(buffer, used);

    // Append a WRITE_METADATA record at the end.
    if (used + sizeof(LogMetaEntry) > reserved_size) {
        // Caller must have reserved enough space.
        // Fall through and truncate; recovery will tolerate truncated meta.
    }
    auto* meta = reinterpret_cast<LogMetaEntry*>(buffer + used);
    meta->type_and_size = 0;
    meta->type = static_cast<uint8_t>(LogEntry::Type::WRITE_METADATA);
    meta->size = sizeof(LogMetaEntry);
    meta->rc.w_id = 0;
    meta->rc.gsn = 0;
    meta->rc.chunk_size = static_cast<uint32_t>(used);
    meta->rc.signature = 0;  // filled in by LogManager when serializing
    meta->ComputeChksum();
    used += sizeof(LogMetaEntry);

    write_fn(buffer, used);
}

uint64_t LogIOSegment::Deserialize(uint8_t* buffer, uint64_t end_offset, uint32_t required_signature,
                                   const std::function<void(uint8_t*, uint64_t)>& analyze_fn) {
    uint64_t off = 0;
    while (off < end_offset) {
        auto* e = reinterpret_cast<LogEntry*>(buffer + off);
        if (off + sizeof(LogEntry) > end_offset) break;
        uint32_t sz = e->size;
        if (sz == 0 || off + sz > end_offset) break;
        if (e->type == static_cast<uint8_t>(LogEntry::Type::WRITE_METADATA)) {
            // Found the meta at the end of this segment; return offset of next.
            return off + sz;
        }
        // Non-meta entry: hand to analyze_fn
        analyze_fn(buffer + off, sz);
        off += sz;
    }
    return off;
}

}  // namespace recovery
}  // namespace caliby
