/**
 * @file btree_index.cpp
 * @brief Implementation of B-tree Index Wrapper for Collection Metadata
 */

#include "btree_index.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace caliby {

//=============================================================================
// BTreeKey Utilities
//=============================================================================

int compare_keys(const BTreeKey& a, const BTreeKey& b) {
    // Different types compare by type index
    if (a.index() != b.index()) {
        return (a.index() < b.index()) ? -1 : 1;
    }
    
    return std::visit([](auto&& av, auto&& bv) -> int {
        using T = std::decay_t<decltype(av)>;
        using U = std::decay_t<decltype(bv)>;
        
        if constexpr (std::is_same_v<T, U>) {
            if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                if (av < bv) return -1;
                if (av > bv) return 1;
                return 0;
            } else if constexpr (std::is_same_v<T, std::string>) {
                return av.compare(bv);
            } else if constexpr (std::is_same_v<T, bool>) {
                if (av == bv) return 0;
                return av ? 1 : -1;
            }
        }
        return 0;
    }, a, b);
}

BTreeKeyType get_key_type(const BTreeKey& key) {
    return std::visit([](auto&& v) -> BTreeKeyType {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, int64_t>) return BTreeKeyType::INT;
        else if constexpr (std::is_same_v<T, double>) return BTreeKeyType::FLOAT;
        else if constexpr (std::is_same_v<T, std::string>) return BTreeKeyType::STRING;
        else if constexpr (std::is_same_v<T, bool>) return BTreeKeyType::BOOL;
        else return BTreeKeyType::INT;
    }, key);
}

std::vector<uint8_t> serialize_key(const BTreeKey& key) {
    std::vector<uint8_t> result;
    
    std::visit([&result](auto&& v) {
        using T = std::decay_t<decltype(v)>;
        
        if constexpr (std::is_same_v<T, int64_t>) {
            // Big-endian encoding for proper byte ordering
            result.resize(sizeof(int64_t));
            int64_t val = v;
            // Convert to unsigned and flip sign bit for proper ordering
            uint64_t uval = static_cast<uint64_t>(val) ^ (1ULL << 63);
            for (int i = 7; i >= 0; --i) {
                result[7 - i] = static_cast<uint8_t>((uval >> (i * 8)) & 0xFF);
            }
        } else if constexpr (std::is_same_v<T, double>) {
            result.resize(sizeof(double));
            double val = v;
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(double));
            // Flip sign bit and conditionally flip other bits for proper ordering
            if (bits & (1ULL << 63)) {
                bits = ~bits;
            } else {
                bits ^= (1ULL << 63);
            }
            for (int i = 7; i >= 0; --i) {
                result[7 - i] = static_cast<uint8_t>((bits >> (i * 8)) & 0xFF);
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            result.assign(v.begin(), v.end());
        } else if constexpr (std::is_same_v<T, bool>) {
            result.push_back(v ? 1 : 0);
        }
    }, key);
    
    return result;
}

BTreeKey deserialize_key(const uint8_t* data, size_t len, BTreeKeyType type) {
    switch (type) {
        case BTreeKeyType::INT: {
            if (len < sizeof(int64_t)) {
                throw std::runtime_error("Invalid key data for INT");
            }
            uint64_t uval = 0;
            for (int i = 0; i < 8; ++i) {
                uval = (uval << 8) | data[i];
            }
            uval ^= (1ULL << 63);
            return static_cast<int64_t>(uval);
        }
        case BTreeKeyType::FLOAT: {
            if (len < sizeof(double)) {
                throw std::runtime_error("Invalid key data for FLOAT");
            }
            uint64_t bits = 0;
            for (int i = 0; i < 8; ++i) {
                bits = (bits << 8) | data[i];
            }
            if (bits & (1ULL << 63)) {
                bits ^= (1ULL << 63);
            } else {
                bits = ~bits;
            }
            double val;
            std::memcpy(&val, &bits, sizeof(double));
            return val;
        }
        case BTreeKeyType::STRING:
            return std::string(reinterpret_cast<const char*>(data), len);
        case BTreeKeyType::BOOL:
            return len > 0 && data[0] != 0;
        default:
            throw std::runtime_error("Unknown key type");
    }
}

//=============================================================================
// Composite Key Utilities
//=============================================================================

int compare_composite_keys(const CompositeKey& a, const CompositeKey& b) {
    size_t min_len = std::min(a.size(), b.size());
    
    for (size_t i = 0; i < min_len; ++i) {
        int cmp = compare_keys(a[i], b[i]);
        if (cmp != 0) return cmp;
    }
    
    // If all compared fields are equal, shorter key is smaller
    if (a.size() < b.size()) return -1;
    if (a.size() > b.size()) return 1;
    return 0;
}

int compare_key_prefix(const CompositeKey& key, const CompositeKey& prefix) {
    if (prefix.empty()) return 0;  // Empty prefix matches everything
    if (key.size() < prefix.size()) return 0;  // Can't have prefix if key is shorter
    
    for (size_t i = 0; i < prefix.size(); ++i) {
        int cmp = compare_keys(key[i], prefix[i]);
        if (cmp != 0) return cmp;
    }
    
    return 0;  // Key has prefix
}

std::vector<uint8_t> serialize_composite_key(const CompositeKey& key) {
    std::vector<uint8_t> result;
    
    // Header: field count (1 byte)
    result.push_back(static_cast<uint8_t>(key.size()));
    
    for (const auto& field : key) {
        // Type (1 byte)
        BTreeKeyType type = get_key_type(field);
        result.push_back(static_cast<uint8_t>(type));
        
        // Serialize field value
        auto field_bytes = serialize_key(field);
        
        // Length (2 bytes, big-endian)
        uint16_t len = static_cast<uint16_t>(field_bytes.size());
        result.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
        result.push_back(static_cast<uint8_t>(len & 0xFF));
        
        // Data
        result.insert(result.end(), field_bytes.begin(), field_bytes.end());
    }
    
    return result;
}

CompositeKey deserialize_composite_key(const uint8_t* data, size_t len) {
    CompositeKey result;
    
    if (len == 0) return result;
    
    size_t offset = 0;
    uint8_t field_count = data[offset++];
    
    for (uint8_t i = 0; i < field_count && offset < len; ++i) {
        // Type
        BTreeKeyType type = static_cast<BTreeKeyType>(data[offset++]);
        
        // Length
        if (offset + 2 > len) break;
        uint16_t field_len = (static_cast<uint16_t>(data[offset]) << 8) | data[offset + 1];
        offset += 2;
        
        // Data
        if (offset + field_len > len) break;
        BTreeKey field = deserialize_key(data + offset, field_len, type);
        result.push_back(std::move(field));
        offset += field_len;
    }
    
    return result;
}

//=============================================================================
// BTreeMetadataIndex Implementation
//=============================================================================

BTreeMetadataIndex::BTreeMetadataIndex(const std::string& field_name,
                                       BTreeKeyType key_type,
                                       bool unique)
    : field_name_(field_name)
    , key_type_(key_type)
    , unique_(unique)
    , btree_(std::make_unique<BTree>())
{
}

BTreeMetadataIndex::~BTreeMetadataIndex() = default;

std::vector<uint8_t> BTreeMetadataIndex::make_composite_key(const BTreeKey& key, uint64_t doc_id) const {
    // For non-unique indexes, append doc_id to key for uniqueness
    auto key_bytes = serialize_key(key);
    
    if (!unique_) {
        // Append doc_id in big-endian for proper ordering
        size_t old_size = key_bytes.size();
        key_bytes.resize(old_size + sizeof(uint64_t));
        for (int i = 7; i >= 0; --i) {
            key_bytes[old_size + (7 - i)] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
        }
    }
    
    return key_bytes;
}

void BTreeMetadataIndex::insert(const BTreeKey& key, uint64_t doc_id) {
    std::unique_lock lock(mutex_);
    
    auto key_bytes = make_composite_key(key, doc_id);
    
    // Payload is just the doc_id for unique indexes, empty for non-unique
    std::vector<uint8_t> payload;
    if (unique_) {
        payload.resize(sizeof(uint64_t));
        for (int i = 7; i >= 0; --i) {
            payload[7 - i] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
        }
    }
    
    btree_->insert(
        std::span<uint8_t>(key_bytes.data(), key_bytes.size()),
        std::span<uint8_t>(payload.data(), payload.size())
    );
    
    entry_count_.fetch_add(1);
}

void BTreeMetadataIndex::remove(const BTreeKey& key, uint64_t doc_id) {
    std::unique_lock lock(mutex_);
    
    auto key_bytes = make_composite_key(key, doc_id);
    
    if (btree_->remove(std::span<uint8_t>(key_bytes.data(), key_bytes.size()))) {
        entry_count_.fetch_sub(1);
    }
}

void BTreeMetadataIndex::insert_batch(const std::vector<std::pair<BTreeKey, uint64_t>>& entries) {
    for (const auto& [key, doc_id] : entries) {
        insert(key, doc_id);
    }
}

std::vector<uint64_t> BTreeMetadataIndex::lookup(const BTreeKey& key) const {
    std::shared_lock lock(mutex_);
    
    std::vector<uint64_t> results;
    
    auto key_bytes = serialize_key(key);
    
    if (unique_) {
        // Direct lookup
        uint8_t payload[sizeof(uint64_t)];
        int len = btree_->lookup(
            std::span<uint8_t>(const_cast<uint8_t*>(key_bytes.data()), key_bytes.size()),
            payload, sizeof(payload)
        );
        
        if (len > 0) {
            uint64_t doc_id = 0;
            for (int i = 0; i < 8; ++i) {
                doc_id = (doc_id << 8) | payload[i];
            }
            results.push_back(doc_id);
        }
    } else {
        // Scan all entries with this key prefix
        // For non-unique, the key has doc_id appended, so we need range scan
        auto scan_key = key_bytes;
        
        const_cast<BTree*>(btree_.get())->scanAsc(
            std::span<uint8_t>(scan_key.data(), scan_key.size()),
            [&](BTreeNode& node, unsigned slot) -> bool {
                // Check if key still matches prefix
                auto node_key = std::span<uint8_t>(node.getKey(slot), node.slot[slot].keyLen);
                
                if (node_key.size() < key_bytes.size()) {
                    return false;  // Stop scan
                }
                
                // Compare prefix
                if (std::memcmp(node_key.data(), key_bytes.data(), key_bytes.size()) != 0) {
                    return false;  // Prefix doesn't match, stop
                }
                
                // Extract doc_id from composite key
                if (node_key.size() >= key_bytes.size() + sizeof(uint64_t)) {
                    uint64_t doc_id = 0;
                    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                        doc_id = (doc_id << 8) | node_key[key_bytes.size() + i];
                    }
                    results.push_back(doc_id);
                }
                
                return true;  // Continue scan
            }
        );
    }
    
    return results;
}

BTreeRangeIterator BTreeMetadataIndex::range_scan(
    const std::optional<BTreeKey>& min_key,
    const std::optional<BTreeKey>& max_key,
    bool include_min,
    bool include_max) const {
    
    std::shared_lock lock(mutex_);
    
    BTreeRangeIterator iter;
    
    // Determine start key
    std::vector<uint8_t> start_key;
    if (min_key) {
        start_key = serialize_key(*min_key);
    }
    
    std::vector<uint8_t> end_key;
    if (max_key) {
        end_key = serialize_key(*max_key);
    }
    
    const_cast<BTree*>(btree_.get())->scanAsc(
        std::span<uint8_t>(start_key.data(), start_key.size()),
        [&](BTreeNode& node, unsigned slot) -> bool {
            auto node_key = std::span<uint8_t>(node.getKey(slot), node.slot[slot].keyLen);
            
            // Extract actual key (without doc_id suffix for non-unique)
            size_t actual_key_len = unique_ ? node_key.size() : 
                (node_key.size() > sizeof(uint64_t) ? node_key.size() - sizeof(uint64_t) : 0);
            
            // Check min bound
            if (min_key && !include_min) {
                auto min_bytes = serialize_key(*min_key);
                if (actual_key_len == min_bytes.size() &&
                    std::memcmp(node_key.data(), min_bytes.data(), min_bytes.size()) == 0) {
                    return true;  // Skip this key, continue
                }
            }
            
            // Check max bound
            if (max_key) {
                auto max_bytes = serialize_key(*max_key);
                int cmp = 0;
                size_t cmp_len = std::min(actual_key_len, max_bytes.size());
                if (cmp_len > 0) {
                    cmp = std::memcmp(node_key.data(), max_bytes.data(), cmp_len);
                }
                if (cmp > 0 || (cmp == 0 && actual_key_len > max_bytes.size())) {
                    return false;  // Past max, stop
                }
                if (cmp == 0 && !include_max) {
                    return false;  // At max but not inclusive, stop
                }
            }
            
            // Extract doc_id
            if (!unique_ && node_key.size() >= sizeof(uint64_t)) {
                uint64_t doc_id = 0;
                size_t doc_id_start = node_key.size() - sizeof(uint64_t);
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    doc_id = (doc_id << 8) | node_key[doc_id_start + i];
                }
                iter.results_.push_back(doc_id);
            } else if (unique_) {
                // Get doc_id from payload
                auto payload = node.getPayload(slot);
                if (payload.size() >= sizeof(uint64_t)) {
                    uint64_t doc_id = 0;
                    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                        doc_id = (doc_id << 8) | payload[i];
                    }
                    iter.results_.push_back(doc_id);
                }
            }
            
            return true;
        }
    );
    
    return iter;
}

std::vector<uint64_t> BTreeMetadataIndex::less_than(const BTreeKey& value, bool inclusive) const {
    return range_scan(std::nullopt, value, true, inclusive).collect();
}

std::vector<uint64_t> BTreeMetadataIndex::greater_than(const BTreeKey& value, bool inclusive) const {
    return range_scan(value, std::nullopt, inclusive, true).collect();
}

bool BTreeMetadataIndex::contains(const BTreeKey& key) const {
    return !lookup(key).empty();
}

//=============================================================================
// DocIdBitmap Implementation
//=============================================================================

DocIdBitmap::DocIdBitmap(size_t capacity) {
    // Pre-allocate some buckets based on expected capacity
    bits_.reserve(capacity / BITS_PER_WORD + 1);
}

void DocIdBitmap::set(uint64_t doc_id) {
    uint64_t block = block_id(doc_id);
    uint64_t offset = bit_offset(doc_id);
    
    auto it = bits_.find(block);
    if (it == bits_.end()) {
        bits_[block] = (1ULL << offset);
        set_count_++;
    } else if (!(it->second & (1ULL << offset))) {
        it->second |= (1ULL << offset);
        set_count_++;
    }
}

void DocIdBitmap::clear(uint64_t doc_id) {
    uint64_t block = block_id(doc_id);
    uint64_t offset = bit_offset(doc_id);
    
    auto it = bits_.find(block);
    if (it != bits_.end() && (it->second & (1ULL << offset))) {
        it->second &= ~(1ULL << offset);
        set_count_--;
        if (it->second == 0) {
            bits_.erase(it);
        }
    }
}

bool DocIdBitmap::test(uint64_t doc_id) const {
    uint64_t block = block_id(doc_id);
    uint64_t offset = bit_offset(doc_id);
    
    auto it = bits_.find(block);
    if (it == bits_.end()) {
        return false;
    }
    return (it->second & (1ULL << offset)) != 0;
}

std::vector<uint64_t> DocIdBitmap::to_vector() const {
    std::vector<uint64_t> result;
    result.reserve(set_count_);
    
    // Collect all blocks and sort by block_id
    std::vector<std::pair<uint64_t, uint64_t>> sorted_blocks(bits_.begin(), bits_.end());
    std::sort(sorted_blocks.begin(), sorted_blocks.end());
    
    for (const auto& [block, word] : sorted_blocks) {
        uint64_t base = block * BITS_PER_WORD;
        for (uint64_t i = 0; i < BITS_PER_WORD; ++i) {
            if (word & (1ULL << i)) {
                result.push_back(base + i);
            }
        }
    }
    
    return result;
}

DocIdBitmap DocIdBitmap::operator&(const DocIdBitmap& other) const {
    DocIdBitmap result;
    
    // Iterate over smaller bitmap
    const auto& smaller = (bits_.size() < other.bits_.size()) ? bits_ : other.bits_;
    const auto& larger = (bits_.size() < other.bits_.size()) ? other.bits_ : bits_;
    
    for (const auto& [block, word] : smaller) {
        auto it = larger.find(block);
        if (it != larger.end()) {
            uint64_t intersection = word & it->second;
            if (intersection) {
                result.bits_[block] = intersection;
                result.set_count_ += __builtin_popcountll(intersection);
            }
        }
    }
    
    return result;
}

DocIdBitmap DocIdBitmap::operator|(const DocIdBitmap& other) const {
    DocIdBitmap result = *this;
    result |= other;
    return result;
}

DocIdBitmap& DocIdBitmap::operator&=(const DocIdBitmap& other) {
    std::vector<uint64_t> to_remove;
    
    for (auto& [block, word] : bits_) {
        auto it = other.bits_.find(block);
        if (it == other.bits_.end()) {
            to_remove.push_back(block);
        } else {
            uint64_t old_count = __builtin_popcountll(word);
            word &= it->second;
            set_count_ -= old_count - __builtin_popcountll(word);
            if (word == 0) {
                to_remove.push_back(block);
            }
        }
    }
    
    for (uint64_t block : to_remove) {
        set_count_ -= __builtin_popcountll(bits_[block]);
        bits_.erase(block);
    }
    
    return *this;
}

DocIdBitmap& DocIdBitmap::operator|=(const DocIdBitmap& other) {
    for (const auto& [block, word] : other.bits_) {
        auto it = bits_.find(block);
        if (it == bits_.end()) {
            bits_[block] = word;
            set_count_ += __builtin_popcountll(word);
        } else {
            uint64_t old_count = __builtin_popcountll(it->second);
            it->second |= word;
            set_count_ += __builtin_popcountll(it->second) - old_count;
        }
    }
    return *this;
}

DocIdBitmap DocIdBitmap::from_vector(const std::vector<uint64_t>& doc_ids) {
    DocIdBitmap result;
    for (uint64_t id : doc_ids) {
        result.set(id);
    }
    return result;
}

//=============================================================================
// CompositeMetadataIndex Implementation
//=============================================================================

CompositeMetadataIndex::CompositeMetadataIndex(
    const std::vector<std::string>& field_names,
    const std::vector<BTreeKeyType>& key_types,
    bool unique)
    : field_names_(field_names)
    , key_types_(key_types)
    , unique_(unique)
    , btree_(std::make_unique<BTree>())
{
    if (field_names.empty()) {
        throw std::runtime_error("CompositeMetadataIndex requires at least one field");
    }
    if (field_names.size() != key_types.size()) {
        throw std::runtime_error("Field names and key types must have same size");
    }
}

CompositeMetadataIndex::~CompositeMetadataIndex() = default;

std::vector<uint8_t> CompositeMetadataIndex::make_internal_key(
    const CompositeKey& key, uint64_t doc_id) const {
    
    auto key_bytes = serialize_composite_key(key);
    
    if (!unique_) {
        // Append doc_id in big-endian for proper ordering
        size_t old_size = key_bytes.size();
        key_bytes.resize(old_size + sizeof(uint64_t));
        for (int i = 7; i >= 0; --i) {
            key_bytes[old_size + (7 - i)] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
        }
    }
    
    return key_bytes;
}

std::vector<uint8_t> CompositeMetadataIndex::make_prefix_lower_bound(
    const CompositeKey& prefix) const {
    
    return serialize_composite_key(prefix);
}

std::vector<uint8_t> CompositeMetadataIndex::make_prefix_upper_bound(
    const CompositeKey& prefix) const {
    
    // For upper bound, we increment the last byte to get the next key
    auto bytes = serialize_composite_key(prefix);
    
    // Append 0xFF bytes to make it larger than any key with this prefix
    bytes.push_back(0xFF);
    bytes.push_back(0xFF);
    bytes.push_back(0xFF);
    bytes.push_back(0xFF);
    
    return bytes;
}

void CompositeMetadataIndex::insert(const CompositeKey& key, uint64_t doc_id) {
    if (key.size() != field_names_.size()) {
        throw std::runtime_error("Key must have " + std::to_string(field_names_.size()) + " fields");
    }
    
    std::unique_lock lock(mutex_);
    
    auto key_bytes = make_internal_key(key, doc_id);
    
    // Payload is just the doc_id for unique indexes, empty for non-unique
    std::vector<uint8_t> payload;
    if (unique_) {
        payload.resize(sizeof(uint64_t));
        for (int i = 7; i >= 0; --i) {
            payload[7 - i] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
        }
    }
    
    btree_->insert(
        std::span<uint8_t>(key_bytes.data(), key_bytes.size()),
        std::span<uint8_t>(payload.data(), payload.size())
    );
    
    entry_count_.fetch_add(1);
}

void CompositeMetadataIndex::remove(const CompositeKey& key, uint64_t doc_id) {
    std::unique_lock lock(mutex_);
    
    auto key_bytes = make_internal_key(key, doc_id);
    
    if (btree_->remove(std::span<uint8_t>(key_bytes.data(), key_bytes.size()))) {
        entry_count_.fetch_sub(1);
    }
}

void CompositeMetadataIndex::insert_batch(
    const std::vector<std::pair<CompositeKey, uint64_t>>& entries) {
    
    for (const auto& [key, doc_id] : entries) {
        insert(key, doc_id);
    }
}

std::vector<uint64_t> CompositeMetadataIndex::lookup(const CompositeKey& key) const {
    if (key.size() != field_names_.size()) {
        throw std::runtime_error("Full key lookup requires all " + 
                                 std::to_string(field_names_.size()) + " fields");
    }
    
    return prefix_lookup(key);
}

std::vector<uint64_t> CompositeMetadataIndex::prefix_lookup(const CompositeKey& prefix) const {
    if (prefix.empty()) {
        throw std::runtime_error("Prefix cannot be empty");
    }
    if (prefix.size() > field_names_.size()) {
        throw std::runtime_error("Prefix has more fields than index");
    }
    
    std::shared_lock lock(mutex_);
    std::vector<uint64_t> results;
    
    auto prefix_bytes = serialize_composite_key(prefix);
    
    const_cast<BTree*>(btree_.get())->scanAsc(
        std::span<uint8_t>(prefix_bytes.data(), prefix_bytes.size()),
        [&](BTreeNode& node, unsigned slot) -> bool {
            auto node_key = std::span<uint8_t>(node.getKey(slot), node.slot[slot].keyLen);
            
            if (node_key.size() < prefix_bytes.size()) {
                return false;  // Stop scan
            }
            
            // Compare prefix (just the composite key part, not doc_id suffix)
            // Check if the serialized key starts with our prefix
            if (std::memcmp(node_key.data(), prefix_bytes.data(), prefix_bytes.size()) != 0) {
                return false;  // Prefix doesn't match, stop
            }
            
            // Extract doc_id from the end of the key (for non-unique)
            // Or from payload (for unique)
            if (unique_) {
                auto payload = node.getPayload(slot);
                if (node.slot[slot].payloadLen >= sizeof(uint64_t)) {
                    uint64_t doc_id = 0;
                    for (int i = 0; i < 8; ++i) {
                        doc_id = (doc_id << 8) | payload[i];
                    }
                    results.push_back(doc_id);
                }
            } else {
                // Doc ID is at the end of the key
                if (node_key.size() >= prefix_bytes.size() + sizeof(uint64_t)) {
                    size_t doc_id_offset = node_key.size() - sizeof(uint64_t);
                    uint64_t doc_id = 0;
                    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                        doc_id = (doc_id << 8) | node_key[doc_id_offset + i];
                    }
                    results.push_back(doc_id);
                }
            }
            
            return true;  // Continue scan
        }
    );
    
    return results;
}

BTreeRangeIterator CompositeMetadataIndex::prefix_range_scan(
    const CompositeKey& prefix,
    const std::optional<BTreeKey>& min_key,
    const std::optional<BTreeKey>& max_key,
    bool include_min,
    bool include_max) const {
    
    std::shared_lock lock(mutex_);
    BTreeRangeIterator iter;
    
    // Build the start key by extending prefix with min_key
    CompositeKey start_prefix = prefix;
    if (min_key) {
        start_prefix.push_back(*min_key);
    }
    auto start_bytes = serialize_composite_key(start_prefix);
    
    // Build the end key by extending prefix with max_key
    CompositeKey end_prefix = prefix;
    if (max_key) {
        end_prefix.push_back(*max_key);
    }
    auto end_bytes = max_key ? serialize_composite_key(end_prefix) : make_prefix_upper_bound(prefix);
    
    const_cast<BTree*>(btree_.get())->scanAsc(
        std::span<uint8_t>(start_bytes.data(), start_bytes.size()),
        [&](BTreeNode& node, unsigned slot) -> bool {
            auto node_key = std::span<uint8_t>(node.getKey(slot), node.slot[slot].keyLen);
            
            // Check if we're still within the prefix
            auto base_prefix_bytes = serialize_composite_key(prefix);
            if (node_key.size() < base_prefix_bytes.size() ||
                std::memcmp(node_key.data(), base_prefix_bytes.data(), base_prefix_bytes.size()) != 0) {
                return false;  // Out of prefix range, stop
            }
            
            // Check max bound
            if (max_key && node_key.size() >= end_bytes.size()) {
                int cmp = std::memcmp(node_key.data(), end_bytes.data(), end_bytes.size());
                if (cmp > 0 || (cmp == 0 && !include_max)) {
                    return false;  // Beyond max, stop
                }
            }
            
            // Check min bound (skip if needed)
            if (min_key && !include_min) {
                if (node_key.size() >= start_bytes.size() &&
                    std::memcmp(node_key.data(), start_bytes.data(), start_bytes.size()) == 0) {
                    return true;  // Skip exact min match
                }
            }
            
            // Extract doc_id
            if (unique_) {
                auto payload = node.getPayload(slot);
                if (node.slot[slot].payloadLen >= sizeof(uint64_t)) {
                    uint64_t doc_id = 0;
                    for (int i = 0; i < 8; ++i) {
                        doc_id = (doc_id << 8) | payload[i];
                    }
                    iter.results_.push_back(doc_id);
                }
            } else {
                if (node_key.size() >= base_prefix_bytes.size() + sizeof(uint64_t)) {
                    size_t doc_id_offset = node_key.size() - sizeof(uint64_t);
                    uint64_t doc_id = 0;
                    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                        doc_id = (doc_id << 8) | node_key[doc_id_offset + i];
                    }
                    iter.results_.push_back(doc_id);
                }
            }
            
            return true;
        }
    );
    
    return iter;
}

bool CompositeMetadataIndex::contains(const CompositeKey& key) const {
    return !lookup(key).empty();
}

//=============================================================================
// DocIdIndex Implementation
//=============================================================================

DocIdIndex::DocIdIndex()
    : btree_(std::make_unique<BTree>())
{
}

// Recovery constructor: reuse existing BTree from slotId
DocIdIndex::DocIdIndex(unsigned btreeSlotId)
    : btree_(std::make_unique<BTree>(btreeSlotId))
{
}

DocIdIndex::~DocIdIndex() = default;

unsigned DocIdIndex::getBTreeSlotId() const {
    return btree_->getSlotId();
}

void DocIdIndex::insert(uint64_t doc_id, const DocLocation& location) {
    std::unique_lock lock(mutex_);
    
    // Serialize doc_id as key (big-endian)
    uint8_t key[sizeof(uint64_t)];
    for (int i = 7; i >= 0; --i) {
        key[7 - i] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
    }
    
    // Serialize location as payload: page_id (8 bytes) + slot (2 bytes) + doc_length (4 bytes)
    uint8_t payload[sizeof(PID) + sizeof(uint16_t) + sizeof(uint32_t)];
    uint64_t pid = location.page_id;
    for (int i = 7; i >= 0; --i) {
        payload[7 - i] = static_cast<uint8_t>((pid >> (i * 8)) & 0xFF);
    }
    payload[8] = static_cast<uint8_t>((location.slot >> 8) & 0xFF);
    payload[9] = static_cast<uint8_t>(location.slot & 0xFF);
    // doc_length in big-endian
    payload[10] = static_cast<uint8_t>((location.doc_length >> 24) & 0xFF);
    payload[11] = static_cast<uint8_t>((location.doc_length >> 16) & 0xFF);
    payload[12] = static_cast<uint8_t>((location.doc_length >> 8) & 0xFF);
    payload[13] = static_cast<uint8_t>(location.doc_length & 0xFF);
    
    btree_->insert(
        std::span<uint8_t>(key, sizeof(key)),
        std::span<uint8_t>(payload, sizeof(payload))
    );
    
    count_.fetch_add(1);
}

std::optional<DocIdIndex::DocLocation> DocIdIndex::lookup(uint64_t doc_id) const {
    std::shared_lock lock(mutex_);
    
    uint8_t key[sizeof(uint64_t)];
    for (int i = 7; i >= 0; --i) {
        key[7 - i] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
    }
    
    uint8_t payload[sizeof(PID) + sizeof(uint16_t) + sizeof(uint32_t)];
    int len = btree_->lookup(
        std::span<uint8_t>(key, sizeof(key)),
        payload, sizeof(payload)
    );
    
    if (len <= 0) {
        return std::nullopt;
    }
    
    // Deserialize location
    PID pid = 0;
    for (int i = 0; i < 8; ++i) {
        pid = (pid << 8) | payload[i];
    }
    uint16_t slot = (static_cast<uint16_t>(payload[8]) << 8) | payload[9];
    
    // Deserialize doc_length (handle old format without doc_length)
    uint32_t doc_length = 0;
    if (len >= 14) {
        doc_length = (static_cast<uint32_t>(payload[10]) << 24) |
                     (static_cast<uint32_t>(payload[11]) << 16) |
                     (static_cast<uint32_t>(payload[12]) << 8) |
                     static_cast<uint32_t>(payload[13]);
    }
    
    return DocLocation{pid, slot, doc_length};
}

void DocIdIndex::update(uint64_t doc_id, const DocLocation& location) {
    // BTree doesn't have direct update, so remove and re-insert
    remove(doc_id);
    
    // Adjust count since remove decremented it
    count_.fetch_add(1);
    
    std::unique_lock lock(mutex_);
    
    uint8_t key[sizeof(uint64_t)];
    for (int i = 7; i >= 0; --i) {
        key[7 - i] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
    }
    
    // Serialize location as payload: page_id (8 bytes) + slot (2 bytes) + doc_length (4 bytes)
    uint8_t payload[sizeof(PID) + sizeof(uint16_t) + sizeof(uint32_t)];
    uint64_t pid = location.page_id;
    for (int i = 7; i >= 0; --i) {
        payload[7 - i] = static_cast<uint8_t>((pid >> (i * 8)) & 0xFF);
    }
    payload[8] = static_cast<uint8_t>((location.slot >> 8) & 0xFF);
    payload[9] = static_cast<uint8_t>(location.slot & 0xFF);
    // doc_length in big-endian
    payload[10] = static_cast<uint8_t>((location.doc_length >> 24) & 0xFF);
    payload[11] = static_cast<uint8_t>((location.doc_length >> 16) & 0xFF);
    payload[12] = static_cast<uint8_t>((location.doc_length >> 8) & 0xFF);
    payload[13] = static_cast<uint8_t>(location.doc_length & 0xFF);
    
    btree_->insert(
        std::span<uint8_t>(key, sizeof(key)),
        std::span<uint8_t>(payload, sizeof(payload))
    );
}

bool DocIdIndex::remove(uint64_t doc_id) {
    std::unique_lock lock(mutex_);
    
    uint8_t key[sizeof(uint64_t)];
    for (int i = 7; i >= 0; --i) {
        key[7 - i] = static_cast<uint8_t>((doc_id >> (i * 8)) & 0xFF);
    }
    
    bool removed = btree_->remove(std::span<uint8_t>(key, sizeof(key)));
    if (removed) {
        count_.fetch_sub(1);
    }
    return removed;
}

bool DocIdIndex::contains(uint64_t doc_id) const {
    return lookup(doc_id).has_value();
}

uint32_t DocIdIndex::get_doc_length(uint64_t doc_id) const {
    auto loc = lookup(doc_id);
    return loc ? loc->doc_length : 0;
}

} // namespace caliby
