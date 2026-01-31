/**
 * @file collection.hpp
 * @brief Caliby Collection System - Document storage with hybrid search
 * 
 * A Collection is a typed document store with optional vector search capabilities.
 * Collections support:
 * - Structured metadata with schema enforcement
 * - Multiple attachable indices (vector, text, B-tree)
 * - Hybrid search combining vector similarity and text relevance
 * - Adaptive filtered search based on selectivity
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <optional>
#include <span>
#include <nlohmann/json.hpp>

#include "calico.hpp"
#include "catalog.hpp"
#include "btree_index.hpp"

// Forward declarations
class BufferManager;

// Forward declarations for index types
template <typename DistanceMetric>
class HNSW;

namespace hnsw_distance {
struct SIMDAcceleratedL2;
struct SIMDAcceleratedIP;
struct SIMDAcceleratedCosine;
}

namespace caliby {

// Forward declare TextIndex
class TextIndex;

//=============================================================================
// Constants
//=============================================================================

constexpr uint64_t COLLECTION_MAGIC = 0xCA11B7C011EC7100ULL;  // "CALIBYCOLLECT"
constexpr uint32_t COLLECTION_VERSION = 1;
constexpr size_t MAX_CONTENT_SIZE = 1024 * 1024;  // 1MB max content
constexpr size_t MAX_METADATA_SIZE = 64 * 1024;   // 64KB max metadata
constexpr size_t INLINE_THRESHOLD = 12 * 1024;    // 12KB inline in page

//=============================================================================
// Schema Types
//=============================================================================

/**
 * Supported field types in collection schema.
 */
enum class FieldType : uint8_t {
    STRING = 0,      // UTF-8 text
    INT = 1,         // 64-bit signed integer
    FLOAT = 2,       // 64-bit float
    BOOL = 3,        // Boolean
    STRING_ARRAY = 4, // Array of strings
    INT_ARRAY = 5,    // Array of integers
};

/**
 * Field definition in a collection schema.
 */
struct FieldDef {
    std::string name;
    FieldType type;
    bool nullable = true;
    
    FieldDef() = default;
    FieldDef(const std::string& n, FieldType t, bool null = true)
        : name(n), type(t), nullable(null) {}
};

/**
 * Collection schema definition.
 */
class Schema {
public:
    Schema() = default;
    
    /**
     * Add a field to the schema.
     */
    void add_field(const std::string& name, FieldType type, bool nullable = true);
    
    /**
     * Get field definition by name.
     */
    const FieldDef* get_field(const std::string& name) const;
    
    /**
     * Get all fields.
     */
    const std::vector<FieldDef>& fields() const { return fields_; }
    
    /**
     * Check if schema has a field.
     */
    bool has_field(const std::string& name) const;
    
    /**
     * Validate metadata against schema.
     */
    bool validate(const nlohmann::json& metadata, std::string& error) const;
    
    /**
     * Serialize schema to JSON.
     */
    nlohmann::json to_json() const;
    
    /**
     * Deserialize schema from JSON.
     */
    static Schema from_json(const nlohmann::json& j);
    
    /**
     * Parse schema from Python-style dict {"field": "type"}.
     */
    static Schema from_dict(const std::unordered_map<std::string, std::string>& dict);
    
private:
    std::vector<FieldDef> fields_;
    std::unordered_map<std::string, size_t> name_to_index_;
};

//=============================================================================
// Metadata Value Type
//=============================================================================

/**
 * Variant type for metadata values.
 */
using MetadataValue = std::variant<
    std::monostate,                    // null
    std::string,                       // string
    int64_t,                           // int
    double,                            // float
    bool,                              // bool
    std::vector<std::string>,          // string[]
    std::vector<int64_t>               // int[]
>;

/**
 * Get FieldType from MetadataValue.
 */
FieldType get_field_type(const MetadataValue& value);

//=============================================================================
// Document
//=============================================================================

/**
 * A document in a collection.
 */
struct Document {
    uint64_t id = 0;                   // Unique document ID
    std::string content;                // Optional text content
    nlohmann::json metadata;            // Typed metadata
    
    Document() = default;
    Document(uint64_t doc_id, const std::string& text = "", 
             const nlohmann::json& meta = nlohmann::json::object())
        : id(doc_id), content(text), metadata(meta) {}
};

/**
 * Search result with document and scores.
 */
struct SearchResult {
    uint64_t doc_id;
    float score;                        // Combined score
    float vector_score = 0.0f;          // Vector similarity score
    float text_score = 0.0f;            // BM25 text score
    std::optional<Document> document;   // Optional full document
};

//=============================================================================
// Filter DSL
//=============================================================================

/**
 * Filter operators.
 */
enum class FilterOp : uint8_t {
    EQ = 0,       // equals
    NE = 1,       // not equals
    GT = 2,       // greater than
    GTE = 3,      // greater than or equal
    LT = 4,       // less than
    LTE = 5,      // less than or equal
    IN = 6,       // in array
    NIN = 7,      // not in array
    CONTAINS = 8, // array contains
    AND = 9,      // logical and
    OR = 10,      // logical or
};

/**
 * A filter condition.
 */
struct FilterCondition {
    FilterOp op;
    std::string field;                  // Field name (for non-logical ops)
    MetadataValue value;                // Comparison value
    std::vector<FilterCondition> children; // For AND/OR operations
    
    FilterCondition() : op(FilterOp::EQ) {}
    FilterCondition(FilterOp o) : op(o) {}
    
    /**
     * Parse filter from JSON DSL.
     * Examples:
     *   {"field": "value"}                    -> EQ
     *   {"field": {"$gt": 10}}                -> GT
     *   {"$and": [{...}, {...}]}              -> AND
     */
    static FilterCondition from_json(const nlohmann::json& j);
    
    /**
     * Evaluate filter against a document's metadata.
     */
    bool evaluate(const nlohmann::json& metadata) const;
};

//=============================================================================
// Index Configuration
//=============================================================================

/**
 * Text index configuration.
 */
struct TextIndexConfig {
    std::vector<std::string> fields = {"content"};  // Fields to index
    std::string analyzer = "standard";               // "standard", "whitespace", "none"
    std::string language = "english";                // For stemming/stopwords
    float k1 = 1.2f;                                // BM25 k1 parameter
    float b = 0.75f;                                // BM25 b parameter
};

/**
 * Metadata index configuration.
 * Supports single-field and composite (multi-field) indices.
 * Composite indices follow the leftmost prefix rule (like MySQL secondary indices).
 */
struct MetadataIndexConfig {
    std::vector<std::string> fields;                // Fields to index (order matters for composite)
    bool unique = false;                            // Unique constraint on the full composite key
    
    // Single-field constructor for convenience
    MetadataIndexConfig() = default;
    explicit MetadataIndexConfig(const std::string& field, bool uniq = false)
        : fields({field}), unique(uniq) {}
    MetadataIndexConfig(std::vector<std::string> flds, bool uniq = false)
        : fields(std::move(flds)), unique(uniq) {}
};

/**
 * B-tree index configuration (legacy alias for backward compatibility).
 * @deprecated Use MetadataIndexConfig instead.
 */
using BTreeIndexConfig = MetadataIndexConfig;

//=============================================================================
// Collection Index Info
//=============================================================================

/**
 * Information about an index attached to a collection.
 */
struct CollectionIndexInfo {
    uint32_t index_id;
    std::string name;
    std::string type;                   // "hnsw", "diskann", "ivfpq", "text", "btree"
    std::string status;                 // "building", "ready", "error"
    nlohmann::json config;              // Index-specific configuration
};

//=============================================================================
// Fusion Strategy
//=============================================================================

/**
 * Score fusion method for hybrid search.
 */
enum class FusionMethod : uint8_t {
    RRF = 0,      // Reciprocal Rank Fusion
    WEIGHTED = 1, // Weighted combination
};

/**
 * Fusion parameters.
 */
struct FusionParams {
    FusionMethod method = FusionMethod::RRF;
    
    // RRF parameter
    int rrf_k = 60;                     // RRF constant (default 60)
    
    // Weighted parameters
    float vector_weight = 0.5f;
    float text_weight = 0.5f;
    bool normalize = true;              // Normalize scores before fusion
};

//=============================================================================
// Distance Metric
//=============================================================================

/**
 * Distance/similarity metric for vector search.
 */
enum class DistanceMetric : uint8_t {
    L2 = 0,       // Euclidean (L2) distance
    COSINE = 1,   // Cosine similarity
    IP = 2,       // Inner product
};

//=============================================================================
// Polymorphic HNSW Wrapper for Runtime Distance Metric Selection
//=============================================================================

/**
 * Abstract base class for HNSW indices with different distance metrics.
 * Allows runtime selection of distance metric while using compile-time
 * optimized SIMD distance functions.
 */
class HNSWIndexBase {
public:
    virtual ~HNSWIndexBase() = default;
    
    // Insert a vector with a specific ID (doc_id used as node_id)
    virtual void addPointWithId(const float* data, uint32_t id) = 0;
    
    // Insert multiple vectors with their IDs in parallel
    virtual void addPointsWithIdsParallel(const std::vector<const float*>& data_ptrs,
                                          const std::vector<uint32_t>& ids,
                                          size_t num_threads = 0) = 0;
    
    // Search for k nearest neighbors
    virtual std::vector<std::pair<float, uint32_t>> searchKnn(
        const float* query, size_t k, size_t ef_search = 100) = 0;
    
    // Get distance metric type
    virtual DistanceMetric metric() const = 0;
    
    // Get whether index was recovered from disk
    virtual bool wasRecovered() const = 0;
};

//=============================================================================
// Collection Metadata Page (On-Disk Format)
//=============================================================================

/**
 * Collection metadata stored in page 0.
 */
struct CollectionMetadataPage {
    bool dirty;                         // Page dirty flag (first byte)
    uint8_t reserved1[7];               // Alignment padding
    
    uint64_t magic;                     // COLLECTION_MAGIC
    uint32_t version;                   // COLLECTION_VERSION
    uint32_t flags;                     // Reserved flags
    
    uint64_t doc_count;                 // Number of documents
    uint64_t next_doc_id;               // Next auto-generated doc ID
    
    PID schema_page;                    // Page ID of schema data
    PID id_index_root;                  // Root page of ID B-tree index (deprecated, use btree_slot_id)
    PID free_list_page;                 // Free page list head
    uint32_t id_index_btree_slot_id;    // BTree slot ID for DocIdIndex recovery
    uint32_t reserved_slot_padding;     // Alignment padding
    PID doc_pages_head;                 // Head of document page chain (slotted pages)
    PID doc_pages_tail;                 // Tail of document page chain (current active page)
    
    uint32_t vector_dim;                // Vector dimensions (0 if no vectors)
    uint8_t distance_metric;            // DistanceMetric enum
    uint8_t reserved2[3];               // Alignment padding
    
    // Inline schema for small schemas (< 3KB)
    uint16_t inline_schema_len;
    char inline_schema[3072];           // JSON schema string
    
    uint8_t reserved3[920];             // Pad to 4KB (reduced for id_index_btree_slot_id)
    
    void initialize() {
        // Note: intentionally NOT setting dirty here - caller should manage that
        // dirty field is at the start of the struct for buffer manager compatibility
        std::memset(reserved1, 0, sizeof(reserved1));
        magic = COLLECTION_MAGIC;
        version = COLLECTION_VERSION;
        flags = 0;
        doc_count = 0;
        next_doc_id = 1;
        schema_page = 0;
        id_index_root = 0;
        free_list_page = 0;
        id_index_btree_slot_id = UINT32_MAX;  // Invalid slot ID means no btree
        reserved_slot_padding = 0;
        doc_pages_head = 0;
        doc_pages_tail = 0;
        vector_dim = 0;
        distance_metric = static_cast<uint8_t>(DistanceMetric::COSINE);
        std::memset(reserved2, 0, sizeof(reserved2));
        inline_schema_len = 0;
        std::memset(inline_schema, 0, sizeof(inline_schema));
        std::memset(reserved3, 0, sizeof(reserved3));
    }
    
    bool is_valid() const {
        return magic == COLLECTION_MAGIC && version == COLLECTION_VERSION;
    }
};

static_assert(sizeof(CollectionMetadataPage) <= 4096, "CollectionMetadataPage must fit in one page");

//=============================================================================
// Document Page (On-Disk Format)
//=============================================================================

/**
 * Document page header.
 */
struct DocumentPageHeader {
    bool dirty;                         // Page dirty flag
    uint8_t flags;                      // Page flags
    uint16_t slot_count;                // Number of slots
    uint16_t free_space;                // Free space in page
    uint16_t free_offset;               // Start of free space
    PID next_page;                      // Next page in chain (0 if none)
    PID prev_page;                      // Previous page in chain (0 if none)
};

/**
 * Slot directory entry.
 */
struct SlotEntry {
    uint16_t offset;                    // Offset from page start
    uint16_t length;                    // Length of record
    uint8_t flags;                      // Slot flags (0x01 = has_overflow)
    uint8_t reserved[3];                // Alignment
    
    static constexpr uint8_t FLAG_OVERFLOW = 0x01;
    static constexpr uint8_t FLAG_DELETED = 0x02;
    
    bool has_overflow() const { return flags & FLAG_OVERFLOW; }
    bool is_deleted() const { return flags & FLAG_DELETED; }
};

/**
 * Document record header (stored in document pages).
 */
struct DocumentRecordHeader {
    uint64_t doc_id;                    // Document ID
    uint32_t total_length;              // Total document size
    uint32_t content_length;            // Content string length
    uint32_t metadata_length;           // Metadata (msgpack) length
    PID overflow_page;                  // Overflow page (0 if none)
    // Followed by: content bytes, then metadata bytes
};

/**
 * Overflow page header.
 */
struct OverflowPageHeader {
    bool dirty;
    uint8_t reserved1[7];
    uint64_t parent_doc_id;             // Document this belongs to
    uint32_t continuation_length;       // Data length in this page
    uint32_t reserved2;
    PID next_overflow;                  // Next overflow page (0 = end)
    // Followed by: continuation data
};

//=============================================================================
// Collection Class
//=============================================================================

/**
 * A Collection is a typed document store with optional vector search capabilities.
 */
class Collection {
public:
    /**
     * Create a new collection.
     * @param name Collection name
     * @param schema Schema definition
     * @param vector_dim Vector dimensions (0 for no vector support)
     * @param distance_metric Distance metric for vector search
     */
    Collection(const std::string& name,
               const Schema& schema,
               uint32_t vector_dim = 0,
               DistanceMetric distance_metric = DistanceMetric::COSINE);
    
    /**
     * Open an existing collection.
     * @param name Collection name
     */
    static std::unique_ptr<Collection> open(const std::string& name);
    
    ~Collection();
    
    // Non-copyable
    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;
    
    //-------------------------------------------------------------------------
    // Basic Operations
    //-------------------------------------------------------------------------
    
    /**
     * Get collection name.
     */
    const std::string& name() const { return name_; }
    
    /**
     * Get collection schema.
     */
    const Schema& schema() const { return schema_; }
    
    /**
     * Get document count.
     */
    uint64_t doc_count() const;
    
    /**
     * Get vector dimensions (0 if no vector support).
     */
    uint32_t vector_dim() const { return vector_dim_; }
    
    /**
     * Check if collection supports vectors.
     */
    bool has_vectors() const { return vector_dim_ > 0; }
    
    //-------------------------------------------------------------------------
    // Document Operations
    //-------------------------------------------------------------------------
    
    /**
     * Add documents to the collection.
     * Doc IDs are auto-assigned sequentially starting from 0.
     * @param contents Text content for each document
     * @param metadatas Metadata for each document
     * @param vectors Optional vectors (shape: n_docs x vector_dim)
     * @return Vector of assigned document IDs
     */
    std::vector<uint64_t> add(const std::vector<std::string>& contents,
                              const std::vector<nlohmann::json>& metadatas,
                              const std::vector<std::vector<float>>& vectors = {});
    
    /**
     * Get documents by ID.
     * @param ids Document IDs to retrieve
     * @return Documents (in order of input IDs)
     */
    std::vector<Document> get(const std::vector<uint64_t>& ids);
    
    /**
     * Get documents matching a filter.
     * @param where Filter condition
     * @param limit Maximum number of documents
     * @param offset Number of documents to skip
     * @return Matching documents
     */
    std::vector<Document> get(const FilterCondition& where,
                              size_t limit = 100,
                              size_t offset = 0);
    
    /**
     * Update document metadata.
     * @param ids Document IDs to update
     * @param metadatas New metadata (partial update)
     */
    void update(const std::vector<uint64_t>& ids,
                const std::vector<nlohmann::json>& metadatas);
    
    /**
     * Delete documents by ID.
     * @param ids Document IDs to delete
     */
    void delete_docs(const std::vector<uint64_t>& ids);
    
    /**
     * Delete documents matching a filter.
     * @param where Filter condition
     * @return Number of documents deleted
     */
    size_t delete_docs(const FilterCondition& where);
    
    //-------------------------------------------------------------------------
    // Index Operations
    //-------------------------------------------------------------------------
    
    /**
     * Create a vector index (HNSW).
     * @param name Index name
     * @param M HNSW M parameter
     * @param ef_construction HNSW ef_construction parameter
     */
    void create_hnsw_index(const std::string& name, size_t M = 16, size_t ef_construction = 200);
    
    /**
     * Create a vector index (DiskANN).
     * @param name Index name
     * @param R Maximum degree
     * @param L Build parameter
     * @param alpha Alpha parameter
     */
    void create_diskann_index(const std::string& name, uint32_t R = 64, uint32_t L = 100, float alpha = 1.2f);
    
    /**
     * Create a text index (BM25).
     * @param name Index name
     * @param config Text index configuration
     */
    void create_text_index(const std::string& name, const TextIndexConfig& config = {});
    
    /**
     * Create a metadata index on one or more fields.
     * Supports composite indices with leftmost prefix rule (like MySQL secondary indices).
     * 
     * Example (single field):
     *   create_metadata_index("year_idx", {"year"});
     * 
     * Example (composite):
     *   create_metadata_index("category_year_idx", {"category", "year"});
     *   // Can efficiently query: category=x, (category=x AND year=y)
     *   // Cannot efficiently query: year=y (leftmost field missing)
     * 
     * @param name Index name
     * @param config Metadata index configuration (fields and unique constraint)
     */
    void create_metadata_index(const std::string& name, const MetadataIndexConfig& config);
    
    /**
     * Create a B-tree index on a metadata field.
     * @deprecated Use create_metadata_index instead. This method exists for backward compatibility.
     * @param name Index name
     * @param config B-tree index configuration
     */
    void create_btree_index(const std::string& name, const BTreeIndexConfig& config) {
        create_metadata_index(name, config);
    }
    
    /**
     * List all indices attached to this collection.
     */
    std::vector<CollectionIndexInfo> list_indices() const;
    
    /**
     * Drop an index.
     * @param name Index name
     */
    void drop_index(const std::string& name);
    
    //-------------------------------------------------------------------------
    // Search Operations
    //-------------------------------------------------------------------------
    
    /**
     * Vector similarity search.
     * @param vector Query vector
     * @param index_name Vector index to use
     * @param k Number of results
     * @param where Optional filter condition
     * @param params Search parameters (index-specific)
     * @return Search results
     */
    std::vector<SearchResult> search_vector(
        const std::vector<float>& vector,
        const std::string& index_name,
        size_t k,
        const std::optional<FilterCondition>& where = std::nullopt,
        const nlohmann::json& params = {});
    
    /**
     * Text search (BM25).
     * @param text Query text
     * @param index_name Text index to use
     * @param k Number of results
     * @param where Optional filter condition
     * @return Search results
     */
    std::vector<SearchResult> search_text(
        const std::string& text,
        const std::string& index_name,
        size_t k,
        const std::optional<FilterCondition>& where = std::nullopt);
    
    /**
     * Hybrid search combining vector and text.
     * @param vector Query vector
     * @param vector_index_name Vector index to use
     * @param text Query text
     * @param text_index_name Text index to use
     * @param k Number of results
     * @param fusion Fusion parameters
     * @param where Optional filter condition
     * @return Search results with fused scores
     */
    std::vector<SearchResult> search_hybrid(
        const std::vector<float>& vector,
        const std::string& vector_index_name,
        const std::string& text,
        const std::string& text_index_name,
        size_t k,
        const FusionParams& fusion = {},
        const std::optional<FilterCondition>& where = std::nullopt);
    
    /**
     * Flush all changes to disk.
     */
    void flush();
    
private:
    // Collection metadata
    std::string name_;
    Schema schema_;
    uint32_t vector_dim_;
    DistanceMetric distance_metric_;
    uint32_t collection_id_;
    
    // Private default constructor for internal use (e.g., open())
    Collection() : vector_dim_(0), distance_metric_(DistanceMetric::COSINE), collection_id_(0) {}
    
    // Buffer manager and file access
    BufferManager* bm_ = nullptr;
    int file_fd_ = -1;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    
    // In-memory state
    std::atomic<uint64_t> doc_count_{0};
    std::atomic<uint64_t> next_doc_id_{0};  // Start from 0 to match HNSW node_ids
    
    // Document page chain (slotted pages)
    PID doc_pages_head_{0};              // First page in chain
    PID doc_pages_tail_{0};              // Current active page with free space
    
    // Persistent B-tree ID index (doc_id -> (page_id, slot))
    std::unique_ptr<DocIdIndex> id_index_;
    
    // Index tracking
    std::unordered_map<std::string, CollectionIndexInfo> indices_;
    
    // Actual index object storage
    // HNSW indices (keyed by index name) - polymorphic for different distance metrics
    // Note: doc_id is used directly as the HNSW node_id for direct mapping
    std::unordered_map<std::string, std::unique_ptr<HNSWIndexBase>> hnsw_indices_;
    
    // Text indices (keyed by index name)
    std::unordered_map<std::string, std::unique_ptr<TextIndex>> text_indices_;
    
    // Internal methods
    void load_metadata();
    void save_metadata();
    void save_text_index_state(const std::string& index_name);
    void save_all_text_index_states();
    PID allocate_page();
    void free_page(PID page_id);
    
    // Document storage methods
    void write_document(const Document& doc);
    Document read_document(uint64_t doc_id);
    void delete_document_internal(uint64_t doc_id);
    
    // ID index methods (B-tree)
    void id_index_insert(uint64_t doc_id, PID page_id, uint16_t slot, uint32_t doc_length = 0);
    void id_index_update_doc_length(uint64_t doc_id, uint32_t doc_length);
    std::optional<std::pair<PID, uint16_t>> id_index_lookup(uint64_t doc_id);
    uint32_t id_index_get_doc_length(uint64_t doc_id) const;
    void id_index_remove(uint64_t doc_id);
    
    // Rebuild ID index from persisted document pages (for recovery)
    void rebuild_id_index();
    
    // Filtered search helpers
    std::vector<uint64_t> evaluate_filter(const FilterCondition& filter);
    float estimate_selectivity(const FilterCondition& filter);
};

} // namespace caliby
