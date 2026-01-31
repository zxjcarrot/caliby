/**
 * @file btree_index.hpp
 * @brief B-tree Index Wrapper for Caliby Collection Metadata Fields
 * 
 * Wraps the existing BTree implementation from calico.hpp for use with
 * collection metadata field indexing.
 * Features:
 * - Support for int, float, string, and bool key types
 * - Range scans for filtering
 * - Posting lists for non-unique keys
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <optional>
#include <functional>
#include <map>

#include "calico.hpp"

namespace caliby {

//=============================================================================
// Key Types for Collection Metadata
//=============================================================================

/**
 * Supported B-tree key types for metadata fields.
 */
enum class BTreeKeyType : uint8_t {
    INT = 0,       // 64-bit signed integer
    FLOAT = 1,     // 64-bit float
    STRING = 2,    // Variable-length string (max 256 bytes)
    BOOL = 3,      // Boolean (stored as 1 byte)
};

/**
 * B-tree key value variant.
 */
using BTreeKey = std::variant<
    int64_t,           // INT
    double,            // FLOAT
    std::string,       // STRING
    bool               // BOOL
>;

/**
 * Composite key for multi-field indices.
 * Keys are compared lexicographically by field order.
 */
using CompositeKey = std::vector<BTreeKey>;

/**
 * Compare two B-tree keys.
 * @return -1 if a < b, 0 if a == b, 1 if a > b
 */
int compare_keys(const BTreeKey& a, const BTreeKey& b);

/**
 * Compare two composite keys lexicographically.
 * Compares field by field until a difference is found.
 * @return -1 if a < b, 0 if a == b, 1 if a > b
 */
int compare_composite_keys(const CompositeKey& a, const CompositeKey& b);

/**
 * Compare composite key against a prefix.
 * @return -1 if key < prefix, 0 if key has prefix, 1 if key > prefix
 */
int compare_key_prefix(const CompositeKey& key, const CompositeKey& prefix);

/**
 * Get key type from BTreeKey variant.
 */
BTreeKeyType get_key_type(const BTreeKey& key);

/**
 * Serialize a BTreeKey to bytes for use with BTree.
 */
std::vector<uint8_t> serialize_key(const BTreeKey& key);

/**
 * Serialize a composite key to bytes.
 * Format: [field_count (1 byte)] [type1] [len1] [data1] [type2] [len2] [data2] ...
 */
std::vector<uint8_t> serialize_composite_key(const CompositeKey& key);

/**
 * Deserialize a BTreeKey from bytes.
 */
BTreeKey deserialize_key(const uint8_t* data, size_t len, BTreeKeyType type);

/**
 * Deserialize a composite key from bytes.
 */
CompositeKey deserialize_composite_key(const uint8_t* data, size_t len);

//=============================================================================
// Range Query Result
//=============================================================================

/**
 * Iterator for range query results.
 */
class BTreeRangeIterator {
public:
    BTreeRangeIterator() = default;
    
    /**
     * Check if there are more results.
     */
    bool has_next() const { return current_pos_ < results_.size(); }
    
    /**
     * Get next document ID.
     */
    uint64_t next() { return results_[current_pos_++]; }
    
    /**
     * Get all remaining results as a vector.
     */
    std::vector<uint64_t> collect() {
        std::vector<uint64_t> result(results_.begin() + current_pos_, results_.end());
        current_pos_ = results_.size();
        return result;
    }
    
    /**
     * Get total count.
     */
    size_t size() const { return results_.size(); }
    
private:
    friend class BTreeMetadataIndex;
    friend class CompositeMetadataIndex;
    
    std::vector<uint64_t> results_;
    size_t current_pos_ = 0;
};

//=============================================================================
// B-tree Metadata Index Class
//=============================================================================

/**
 * B-tree index wrapper for metadata field filtering.
 * Uses the existing BTree implementation from calico.hpp.
 */
class BTreeMetadataIndex {
public:
    /**
     * Create a new B-tree metadata index.
     * @param field_name Name of indexed field
     * @param key_type Type of key values
     * @param unique Whether keys must be unique
     */
    BTreeMetadataIndex(const std::string& field_name,
                       BTreeKeyType key_type,
                       bool unique = false);
    
    ~BTreeMetadataIndex();
    
    // Non-copyable
    BTreeMetadataIndex(const BTreeMetadataIndex&) = delete;
    BTreeMetadataIndex& operator=(const BTreeMetadataIndex&) = delete;
    
    //-------------------------------------------------------------------------
    // Index Operations
    //-------------------------------------------------------------------------
    
    /**
     * Insert a key-value pair.
     * @param key The key value
     * @param doc_id Document ID to associate
     */
    void insert(const BTreeKey& key, uint64_t doc_id);
    
    /**
     * Remove a key-value pair.
     * @param key The key value
     * @param doc_id Document ID to remove
     */
    void remove(const BTreeKey& key, uint64_t doc_id);
    
    /**
     * Batch insert.
     * @param entries Vector of (key, doc_id) pairs
     */
    void insert_batch(const std::vector<std::pair<BTreeKey, uint64_t>>& entries);
    
    //-------------------------------------------------------------------------
    // Lookup Operations
    //-------------------------------------------------------------------------
    
    /**
     * Exact lookup: find all doc IDs with given key.
     * @param key The key to search for
     * @return Vector of matching document IDs
     */
    std::vector<uint64_t> lookup(const BTreeKey& key) const;
    
    /**
     * Range scan: find all doc IDs with keys in range.
     * @param min_key Lower bound (inclusive if include_min)
     * @param max_key Upper bound (inclusive if include_max)
     * @param include_min Include lower bound
     * @param include_max Include upper bound
     * @return Range iterator
     */
    BTreeRangeIterator range_scan(
        const std::optional<BTreeKey>& min_key,
        const std::optional<BTreeKey>& max_key,
        bool include_min = true,
        bool include_max = true) const;
    
    /**
     * Less than: find all doc IDs with key < value.
     */
    std::vector<uint64_t> less_than(const BTreeKey& value, bool inclusive = false) const;
    
    /**
     * Greater than: find all doc IDs with key > value.
     */
    std::vector<uint64_t> greater_than(const BTreeKey& value, bool inclusive = false) const;
    
    /**
     * Check if a key exists.
     */
    bool contains(const BTreeKey& key) const;
    
    //-------------------------------------------------------------------------
    // Statistics
    //-------------------------------------------------------------------------
    
    /**
     * Get number of entries.
     */
    uint64_t entry_count() const { return entry_count_.load(); }
    
    /**
     * Get indexed field name.
     */
    const std::string& field_name() const { return field_name_; }
    
    /**
     * Get key type.
     */
    BTreeKeyType key_type() const { return key_type_; }
    
    /**
     * Check if unique constraint.
     */
    bool is_unique() const { return unique_; }
    
private:
    std::string field_name_;
    BTreeKeyType key_type_;
    bool unique_;
    
    // Underlying B-tree from calico.hpp
    std::unique_ptr<BTree> btree_;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    
    // Statistics
    std::atomic<uint64_t> entry_count_{0};
    
    // Helper to make composite key (key + doc_id) for non-unique indexes
    std::vector<uint8_t> make_composite_key(const BTreeKey& key, uint64_t doc_id) const;
};

//=============================================================================
// Composite Metadata Index Class (Multi-Field Index)
//=============================================================================

/**
 * Composite B-tree metadata index for multi-field filtering.
 * Supports the leftmost prefix rule for efficient queries.
 * 
 * Example:
 *   Index on (category, year, author) can efficiently answer:
 *   - category = 'tech'                     (1-field prefix)
 *   - category = 'tech' AND year = 2024     (2-field prefix)
 *   - category = 'tech' AND year = 2024 AND author = 'alice'  (full key)
 *   
 *   Cannot efficiently answer:
 *   - year = 2024                           (leftmost field missing)
 *   - category = 'tech' AND author = 'alice' (middle field skipped)
 */
class CompositeMetadataIndex {
public:
    /**
     * Create a new composite metadata index.
     * @param field_names Names of indexed fields (order matters)
     * @param key_types Types of key values for each field
     * @param unique Whether the full composite key must be unique
     */
    CompositeMetadataIndex(const std::vector<std::string>& field_names,
                           const std::vector<BTreeKeyType>& key_types,
                           bool unique = false);
    
    ~CompositeMetadataIndex();
    
    // Non-copyable
    CompositeMetadataIndex(const CompositeMetadataIndex&) = delete;
    CompositeMetadataIndex& operator=(const CompositeMetadataIndex&) = delete;
    
    //-------------------------------------------------------------------------
    // Index Operations
    //-------------------------------------------------------------------------
    
    /**
     * Insert a composite key-value pair.
     * @param key The composite key (values for each field in order)
     * @param doc_id Document ID to associate
     */
    void insert(const CompositeKey& key, uint64_t doc_id);
    
    /**
     * Remove a composite key-value pair.
     * @param key The composite key
     * @param doc_id Document ID to remove
     */
    void remove(const CompositeKey& key, uint64_t doc_id);
    
    /**
     * Batch insert.
     * @param entries Vector of (composite_key, doc_id) pairs
     */
    void insert_batch(const std::vector<std::pair<CompositeKey, uint64_t>>& entries);
    
    //-------------------------------------------------------------------------
    // Lookup Operations
    //-------------------------------------------------------------------------
    
    /**
     * Exact lookup: find all doc IDs with given composite key.
     * @param key The full composite key to search for
     * @return Vector of matching document IDs
     */
    std::vector<uint64_t> lookup(const CompositeKey& key) const;
    
    /**
     * Prefix lookup: find all doc IDs matching a key prefix.
     * Follows the leftmost prefix rule.
     * @param prefix The prefix keys (must start from first field)
     * @return Vector of matching document IDs
     */
    std::vector<uint64_t> prefix_lookup(const CompositeKey& prefix) const;
    
    /**
     * Prefix range scan: find doc IDs matching prefix with range on next field.
     * Example: prefix = ["tech"], then range scan on "year" field.
     * @param prefix Equality prefix (first N-1 fields)
     * @param min_key Lower bound for Nth field (std::nullopt for no lower bound)
     * @param max_key Upper bound for Nth field (std::nullopt for no upper bound)
     * @param include_min Include lower bound
     * @param include_max Include upper bound
     * @return Range iterator
     */
    BTreeRangeIterator prefix_range_scan(
        const CompositeKey& prefix,
        const std::optional<BTreeKey>& min_key,
        const std::optional<BTreeKey>& max_key,
        bool include_min = true,
        bool include_max = true) const;
    
    /**
     * Check if a composite key exists.
     */
    bool contains(const CompositeKey& key) const;
    
    //-------------------------------------------------------------------------
    // Statistics & Metadata
    //-------------------------------------------------------------------------
    
    /**
     * Get number of entries.
     */
    uint64_t entry_count() const { return entry_count_.load(); }
    
    /**
     * Get indexed field names.
     */
    const std::vector<std::string>& field_names() const { return field_names_; }
    
    /**
     * Get number of fields in composite key.
     */
    size_t field_count() const { return field_names_.size(); }
    
    /**
     * Get key types for each field.
     */
    const std::vector<BTreeKeyType>& key_types() const { return key_types_; }
    
    /**
     * Check if unique constraint.
     */
    bool is_unique() const { return unique_; }
    
    /**
     * Check if a prefix length is valid (leftmost prefix rule).
     */
    bool is_valid_prefix_length(size_t len) const { return len > 0 && len <= field_names_.size(); }
    
private:
    std::vector<std::string> field_names_;
    std::vector<BTreeKeyType> key_types_;
    bool unique_;
    
    // Underlying B-tree from calico.hpp
    std::unique_ptr<BTree> btree_;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    
    // Statistics
    std::atomic<uint64_t> entry_count_{0};
    
    // Helper to make internal key (composite_key + doc_id) for non-unique indexes
    std::vector<uint8_t> make_internal_key(const CompositeKey& key, uint64_t doc_id) const;
    
    // Helper to make prefix boundary keys for range scans
    std::vector<uint8_t> make_prefix_lower_bound(const CompositeKey& prefix) const;
    std::vector<uint8_t> make_prefix_upper_bound(const CompositeKey& prefix) const;
};

//=============================================================================
// Bitmap Operations for Filter Evaluation
//=============================================================================

/**
 * Bitmap for document ID filtering.
 * Uses a sparse representation for efficiency with large doc_id ranges.
 */
class DocIdBitmap {
public:
    DocIdBitmap() = default;
    explicit DocIdBitmap(size_t capacity);
    
    /**
     * Set a bit.
     */
    void set(uint64_t doc_id);
    
    /**
     * Clear a bit.
     */
    void clear(uint64_t doc_id);
    
    /**
     * Check if a bit is set.
     */
    bool test(uint64_t doc_id) const;
    
    /**
     * Get count of set bits.
     */
    size_t count() const { return set_count_; }
    
    /**
     * Check if empty.
     */
    bool empty() const { return set_count_ == 0; }
    
    /**
     * Get all set doc IDs.
     */
    std::vector<uint64_t> to_vector() const;
    
    /**
     * Bitwise AND with another bitmap.
     */
    DocIdBitmap operator&(const DocIdBitmap& other) const;
    
    /**
     * Bitwise OR with another bitmap.
     */
    DocIdBitmap operator|(const DocIdBitmap& other) const;
    
    /**
     * In-place AND.
     */
    DocIdBitmap& operator&=(const DocIdBitmap& other);
    
    /**
     * In-place OR.
     */
    DocIdBitmap& operator|=(const DocIdBitmap& other);
    
    /**
     * Create from vector of doc IDs.
     */
    static DocIdBitmap from_vector(const std::vector<uint64_t>& doc_ids);
    
private:
    // Sparse bitmap using unordered_map for large ranges
    std::unordered_map<uint64_t, uint64_t> bits_;  // block_id -> 64-bit word
    size_t set_count_ = 0;
    static constexpr size_t BITS_PER_WORD = 64;
    
    uint64_t block_id(uint64_t doc_id) const { return doc_id / BITS_PER_WORD; }
    uint64_t bit_offset(uint64_t doc_id) const { return doc_id % BITS_PER_WORD; }
};

//=============================================================================
// ID Index for Documents
//=============================================================================

/**
 * Specialized B-tree index for document ID -> (page_id, slot, doc_length) mapping.
 * Uses the existing BTree from calico.hpp.
 */
class DocIdIndex {
public:
    /**
     * Document location in storage, including document length for BM25 scoring.
     */
    struct DocLocation {
        PID page_id;
        uint16_t slot;
        uint32_t doc_length;  // Document length (word count) for BM25 scoring
        
        DocLocation() : page_id(0), slot(0), doc_length(0) {}
        DocLocation(PID pid, uint16_t s, uint32_t len = 0) 
            : page_id(pid), slot(s), doc_length(len) {}
    };
    
    DocIdIndex();                                  // Create new index
    explicit DocIdIndex(unsigned btreeSlotId);    // Recover from existing BTree slotId
    ~DocIdIndex();
    
    // Non-copyable
    DocIdIndex(const DocIdIndex&) = delete;
    DocIdIndex& operator=(const DocIdIndex&) = delete;
    
    /**
     * Get the underlying BTree's slotId for persistence.
     */
    unsigned getBTreeSlotId() const;
    
    /**
     * Insert a document ID -> location mapping.
     */
    void insert(uint64_t doc_id, const DocLocation& location);
    
    /**
     * Lookup a document location by ID.
     * @return Location if found, nullopt otherwise
     */
    std::optional<DocLocation> lookup(uint64_t doc_id) const;
    
    /**
     * Update a document location.
     */
    void update(uint64_t doc_id, const DocLocation& location);
    
    /**
     * Remove a document ID mapping.
     */
    bool remove(uint64_t doc_id);
    
    /**
     * Check if a document ID exists.
     */
    bool contains(uint64_t doc_id) const;
    
    /**
     * Get count of documents.
     */
    uint64_t count() const { return count_.load(); }
    
    /**
     * Get document length for BM25 scoring.
     * @return Document length, or 0 if not found
     */
    uint32_t get_doc_length(uint64_t doc_id) const;
    
private:
    std::unique_ptr<BTree> btree_;
    mutable std::shared_mutex mutex_;
    std::atomic<uint64_t> count_{0};
};

} // namespace caliby
