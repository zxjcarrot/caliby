/**
 * @file catalog.hpp
 * @brief Caliby Index Catalog System
 * 
 * Provides multi-index management with translation path caching.
 * Each index gets its own file while sharing the buffer pool.
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
#include <vector>

// Forward declaration of global BufferManager (from calico.hpp)
class BufferManager;

namespace caliby {

// Forward declarations
class IndexHandle;
class IndexCatalog;

//=============================================================================
// Constants
//=============================================================================

constexpr uint64_t CATALOG_MAGIC = 0xCA11B7CA7A106000ULL;  // "CALIBYCATALOG"
constexpr uint32_t CATALOG_VERSION = 1;
constexpr uint32_t CATALOG_INDEX_ID = 0;  // Reserved for catalog file

// Page ID bit layout: [index_id (32 bits)][local_page_id (32 bits)]
// This matches TwoLevelPageStateArray::makePID() layout
constexpr uint32_t INDEX_ID_BITS = 32;
constexpr uint32_t LOCAL_PAGE_BITS = 32;
constexpr uint64_t LOCAL_PAGE_MASK = 0xFFFFFFFFULL;
constexpr uint32_t MAX_INDEX_ID = 65535;  // Practical limit (matches TwoLevelPageStateArray::MAX_INDEXES)

constexpr size_t MAX_INDEX_NAME_LEN = 256;
constexpr size_t MAX_FILE_PATH_LEN = 512;
constexpr size_t TYPE_METADATA_SIZE = 256;

//=============================================================================
// Type-Specific Metadata Structures
//=============================================================================

/**
 * HNSW-specific metadata stored in IndexEntry::type_metadata
 */
struct HnswTypeMetadata {
    uint32_t M;              // Number of bidirectional links
    uint32_t ef_construction; // Size of dynamic candidate list during construction
    uint8_t reserved[248];   // Pad to TYPE_METADATA_SIZE
    
    void initialize(uint32_t m, uint32_t ef) {
        M = m;
        ef_construction = ef;
        std::memset(reserved, 0, sizeof(reserved));
    }
};
static_assert(sizeof(HnswTypeMetadata) == TYPE_METADATA_SIZE, "HnswTypeMetadata must match TYPE_METADATA_SIZE");

/**
 * Text index metadata stored in IndexEntry::type_metadata
 */
struct TextTypeMetadata {
    static constexpr size_t MAX_ANALYZER_LEN = 32;
    static constexpr size_t MAX_LANGUAGE_LEN = 32;
    static constexpr uint32_t INVALID_BTREE_SLOT = UINT32_MAX;  // Sentinel for uninitialized
    
    char analyzer[MAX_ANALYZER_LEN];   // "standard", "whitespace", "none"  (32 bytes)
    char language[MAX_LANGUAGE_LEN];   // For stemming/stopwords            (32 bytes)
    float k1;                          // BM25 k1 parameter                 (4 bytes)
    float b;                           // BM25 b parameter                  (4 bytes)
    
    // Persistent BTree state for text index recovery
    uint32_t btree_slot_id;            // BTree slot ID for term dictionary (4 bytes)
    uint32_t reserved_padding;         // Alignment padding                 (4 bytes)
    uint64_t vocab_size;               // Number of unique terms            (8 bytes)
    uint64_t doc_count;                // Number of indexed documents       (8 bytes)
    uint64_t total_doc_length;         // Total of all document lengths     (8 bytes)
    
    // Total used: 32 + 32 + 4 + 4 + 4 + 4 + 8 + 8 + 8 = 104 bytes
    // Reserved: 256 - 104 = 152 bytes
    uint8_t reserved[152];
    
    void initialize(const std::string& ana, const std::string& lang, float k1_val, float b_val) {
        std::memset(this, 0, sizeof(*this));
        std::strncpy(analyzer, ana.c_str(), MAX_ANALYZER_LEN - 1);
        std::strncpy(language, lang.c_str(), MAX_LANGUAGE_LEN - 1);
        k1 = k1_val;
        b = b_val;
        btree_slot_id = INVALID_BTREE_SLOT;  // Not initialized
        reserved_padding = 0;
        vocab_size = 0;
        doc_count = 0;
        total_doc_length = 0;
    }
    
    bool has_valid_btree() const {
        return btree_slot_id != INVALID_BTREE_SLOT;
    }
};
static_assert(sizeof(TextTypeMetadata) == TYPE_METADATA_SIZE, "TextTypeMetadata must match TYPE_METADATA_SIZE");

/**
 * Collection metadata stored in IndexEntry::type_metadata
 * Stores the root PID of the ID B-tree index and other collection state.
 */
struct CollectionTypeMetadata {
    uint64_t id_btree_root_pid;    // Root PID of the ID B-tree index (0 if not yet created)
    uint32_t vector_dim;           // Vector dimensions (0 = no vectors)
    uint32_t distance_metric;      // Distance metric enum value
    uint64_t doc_count;            // Document count (for quick recovery)
    uint64_t next_doc_id;          // Next document ID
    uint8_t reserved[256 - 32];    // Reserved for future use
    
    void initialize(uint32_t vec_dim, uint32_t dist_metric) {
        std::memset(this, 0, sizeof(*this));
        id_btree_root_pid = 0;
        vector_dim = vec_dim;
        distance_metric = dist_metric;
        doc_count = 0;
        next_doc_id = 1;
    }
};
static_assert(sizeof(CollectionTypeMetadata) == TYPE_METADATA_SIZE, "CollectionTypeMetadata must match TYPE_METADATA_SIZE");

/**
 * BTree/Metadata index metadata stored in IndexEntry::type_metadata
 */
struct BTreeTypeMetadata {
    static constexpr size_t MAX_FIELD_LEN = 64;
    static constexpr size_t MAX_FIELDS = 3;  // Support up to 3 fields in composite index
    
    char fields[MAX_FIELDS][MAX_FIELD_LEN];  // Field names
    uint8_t num_fields;                       // Number of fields (1-3)
    bool unique;                              // Unique constraint
    uint8_t reserved[256 - (MAX_FIELDS * MAX_FIELD_LEN) - 2];
    
    void initialize(const std::vector<std::string>& field_list, bool is_unique) {
        std::memset(this, 0, sizeof(*this));
        num_fields = static_cast<uint8_t>(std::min(field_list.size(), static_cast<size_t>(MAX_FIELDS)));
        for (size_t i = 0; i < num_fields; ++i) {
            std::strncpy(fields[i], field_list[i].c_str(), MAX_FIELD_LEN - 1);
        }
        unique = is_unique;
    }
    
    std::vector<std::string> get_fields() const {
        std::vector<std::string> result;
        for (size_t i = 0; i < num_fields; ++i) {
            result.push_back(fields[i]);
        }
        return result;
    }
};
static_assert(sizeof(BTreeTypeMetadata) == TYPE_METADATA_SIZE, "BTreeTypeMetadata must match TYPE_METADATA_SIZE");

//=============================================================================
// Enums
//=============================================================================

enum class IndexType : uint32_t {
    CATALOG = 0,    // Reserved for catalog metadata
    HNSW = 1,       // HNSW index
    DISKANN = 2,    // DiskANN/Vamana index
    IVF = 3,        // IVF index
    COLLECTION = 4, // Collection document storage
    TEXT = 5,       // BM25 text index
    BTREE = 6,      // B-tree metadata index
};

enum class IndexStatus : uint32_t {
    INVALID = 0,
    CREATING = 1,   // Index being created
    ACTIVE = 2,     // Index ready for use
    DELETED = 3,    // Marked for deletion
};

//=============================================================================
// Page ID Utilities
//=============================================================================

/**
 * Compose a global page ID from index_id and local_page_id.
 */
inline uint64_t make_global_page_id(uint32_t index_id, uint64_t local_page_id) {
    return (static_cast<uint64_t>(index_id) << LOCAL_PAGE_BITS) | 
           (local_page_id & LOCAL_PAGE_MASK);
}

/**
 * Extract index_id from a global page ID.
 */
inline uint32_t get_index_id(uint64_t global_page_id) {
    return static_cast<uint32_t>(global_page_id >> LOCAL_PAGE_BITS);
}

/**
 * Extract local_page_id from a global page ID.
 */
inline uint64_t get_local_page_id(uint64_t global_page_id) {
    return global_page_id & LOCAL_PAGE_MASK;
}

//=============================================================================
// Catalog Structures (On-Disk Format)
//=============================================================================

/**
 * Catalog file header (stored in page 0).
 */
struct CatalogHeader {
    uint64_t magic;              // CATALOG_MAGIC
    uint32_t version;            // CATALOG_VERSION
    uint32_t num_indexes;        // Number of active indexes
    uint32_t next_index_id;      // Next available index_id
    uint32_t flags;              // Reserved flags
    uint64_t checksum;           // Header checksum (CRC64 or similar)
    uint8_t reserved[4048];      // Pad to ~4KB
    
    bool is_valid() const {
        return magic == CATALOG_MAGIC && version == CATALOG_VERSION;
    }
    
    void initialize() {
        magic = CATALOG_MAGIC;
        version = CATALOG_VERSION;
        num_indexes = 0;
        next_index_id = 1;  // 0 is reserved for catalog
        flags = 0;
        checksum = 0;
        std::memset(reserved, 0, sizeof(reserved));
    }
};

static_assert(sizeof(CatalogHeader) <= 4096, "CatalogHeader must fit in one page");

/**
 * Per-index metadata entry.
 */
struct IndexEntry {
    uint32_t index_id;           // Unique index identifier
    IndexType index_type;        // Type of index
    IndexStatus status;          // Current status
    uint32_t dimensions;         // Vector dimensions
    uint64_t max_elements;       // Maximum capacity
    uint64_t num_elements;       // Current element count
    uint64_t alloc_pages;        // Number of pages allocated (for recovery)
    uint64_t create_time;        // Creation timestamp (Unix epoch)
    uint64_t modify_time;        // Last modification timestamp
    char name[MAX_INDEX_NAME_LEN];       // Index name (null-terminated)
    char file_path[MAX_FILE_PATH_LEN];   // Backing file path
    uint8_t type_metadata[TYPE_METADATA_SIZE];  // Type-specific config
    
    bool is_active() const { return status == IndexStatus::ACTIVE; }
    bool is_valid() const { return status != IndexStatus::INVALID; }
    
    void clear() {
        std::memset(this, 0, sizeof(IndexEntry));
    }
};

// Calculate entries per page
constexpr size_t ENTRIES_PER_PAGE = 4096 / sizeof(IndexEntry);

/**
 * Page containing index entries.
 */
struct IndexEntryPage {
    IndexEntry entries[ENTRIES_PER_PAGE];
    
    static constexpr size_t capacity() { return ENTRIES_PER_PAGE; }
};

static_assert(sizeof(IndexEntryPage) <= 4096, "IndexEntryPage must fit in one page");

//=============================================================================
// Index Configuration
//=============================================================================

/**
 * HNSW-specific configuration.
 */
struct HNSWConfig {
    uint32_t M = 16;
    uint32_t ef_construction = 100;
    uint32_t max_level = 0;  // 0 = auto-calculate
    bool enable_prefetch = true;
};

/**
 * DiskANN-specific configuration.
 */
struct DiskANNConfig {
    uint32_t R_max_degree = 64;
    float alpha = 1.2f;
    uint32_t L_build = 100;
    bool is_dynamic = false;
};

/**
 * Union of index-specific configurations.
 */
struct IndexConfig {
    uint32_t dimensions = 0;
    uint64_t max_elements = 0;
    
    union {
        HNSWConfig hnsw;
        DiskANNConfig diskann;
    };
    
    IndexConfig() : hnsw{} {}
};

//=============================================================================
// Index Info (Read-Only View)
//=============================================================================

/**
 * Read-only view of index metadata for listing.
 */
struct IndexInfo {
    uint32_t index_id;
    std::string name;
    IndexType type;
    IndexStatus status;
    uint32_t dimensions;
    uint64_t max_elements;
    uint64_t num_elements;
    uint64_t create_time;
    uint64_t modify_time;
    std::string file_path;
};

//=============================================================================
// Index Handle
//=============================================================================

/**
 * Handle to an open index with translation path caching.
 * 
 * The handle caches the file descriptor and translation table pointer
 * to optimize repeated page accesses within the same index.
 */
class IndexHandle {
public:
    IndexHandle() = default;
    IndexHandle(IndexCatalog* catalog, uint32_t index_id, int file_fd,
                const std::string& name, IndexType type, uint32_t dimensions,
                uint64_t max_elements);
    
    // Move-only
    IndexHandle(IndexHandle&& other) noexcept;
    IndexHandle& operator=(IndexHandle&& other) noexcept;
    IndexHandle(const IndexHandle&) = delete;
    IndexHandle& operator=(const IndexHandle&) = delete;
    
    ~IndexHandle();
    
    /**
     * Check if handle is valid.
     */
    bool is_valid() const { return catalog_ != nullptr && index_id_ > 0; }
    
    /**
     * Get global page ID from local page ID.
     * Uses cached index_id for efficiency.
     */
    uint64_t global_page_id(uint64_t local_page_id) const {
        return make_global_page_id(index_id_, local_page_id);
    }
    
    // Accessors
    uint32_t index_id() const { return index_id_; }
    int file_fd() const { return file_fd_; }
    const std::string& name() const { return name_; }
    IndexType type() const { return type_; }
    uint32_t dimensions() const { return dimensions_; }
    uint64_t max_elements() const { return max_elements_; }
    
    /**
     * Get the buffer manager for this index.
     */
    BufferManager* buffer_manager() const;
    
    /**
     * Allocate a new page for this index.
     * Returns the local page ID.
     */
    uint64_t allocate_page();
    
    /**
     * Update the element count in the catalog.
     */
    void update_element_count(uint64_t count);
    
    /**
     * Flush all dirty pages for this index.
     */
    void flush();
    
private:
    IndexCatalog* catalog_ = nullptr;
    uint32_t index_id_ = 0;
    int file_fd_ = -1;
    std::string name_;
    IndexType type_ = IndexType::CATALOG;
    uint32_t dimensions_ = 0;
    uint64_t max_elements_ = 0;
    
    // Translation path cache (for future optimization)
    // void* cached_translation_table_ = nullptr;
};

//=============================================================================
// Multi-File Storage
//=============================================================================

/**
 * Manages file descriptors for multiple index files.
 */
class MultiFileStorage {
public:
    MultiFileStorage() = default;
    ~MultiFileStorage();
    
    // Non-copyable
    MultiFileStorage(const MultiFileStorage&) = delete;
    MultiFileStorage& operator=(const MultiFileStorage&) = delete;
    
    /**
     * Initialize storage with a data directory.
     */
    void initialize(const std::string& data_dir);
    
    /**
     * Open or create a file for an index.
     * @param index_id The index identifier
     * @param filename The filename (relative to data_dir)
     * @param create If true, create file if not exists
     * @return File descriptor, or -1 on error
     */
    int open_file(uint32_t index_id, const std::string& filename, bool create = false);
    
    /**
     * Close a file for an index.
     */
    void close_file(uint32_t index_id);
    
    /**
     * Get file descriptor for an index.
     * @return File descriptor, or -1 if not open
     */
    int get_fd(uint32_t index_id) const;
    
    /**
     * Read a page from disk.
     * @param index_id The index owning the page
     * @param local_page_id Page number within the index file
     * @param buffer Destination buffer (must be page-aligned)
     */
    void read_page(uint32_t index_id, uint64_t local_page_id, void* buffer);
    
    /**
     * Write a page to disk.
     * @param index_id The index owning the page
     * @param local_page_id Page number within the index file
     * @param buffer Source buffer (must be page-aligned)
     */
    void write_page(uint32_t index_id, uint64_t local_page_id, const void* buffer);
    
    /**
     * Get the data directory path.
     */
    const std::string& data_dir() const { return data_dir_; }
    
    /**
     * Generate filename for an index.
     */
    std::string make_index_filename(IndexType type, uint32_t index_id, const std::string& index_name) const;
    
    /**
     * Delete an index file.
     */
    bool delete_file(const std::string& filename);
    
private:
    std::string data_dir_;
    mutable std::shared_mutex fd_mutex_;
    std::unordered_map<uint32_t, int> index_to_fd_;
    
    static constexpr size_t PAGE_SIZE = 4096;
};

//=============================================================================
// Index Catalog
//=============================================================================

/**
 * Central catalog managing all indexes in the system.
 * 
 * The catalog maintains:
 * - A catalog file with index metadata (managed through buffer pool)
 * - File descriptors for each index file
 * - Name-to-index mapping for lookups
 * 
 * Thread-safety: All public methods are thread-safe.
 */
class IndexCatalog {
public:
    /**
     * Get the singleton instance.
     */
    static IndexCatalog& instance();
    
    // Non-copyable singleton
    IndexCatalog(const IndexCatalog&) = delete;
    IndexCatalog& operator=(const IndexCatalog&) = delete;
    
    /**
     * Initialize the catalog in a directory.
     * Creates catalog file if it doesn't exist.
     * @param data_dir Directory to store index files
     * @param cleanup_if_exist If true, removes all existing data before initializing
     */
    void initialize(const std::string& data_dir, bool cleanup_if_exist = false);
    
    /**
     * Check if catalog is initialized.
     */
    bool is_initialized() const { return initialized_.load(); }
    
    /**
     * Shutdown the catalog and close all files.
     */
    void shutdown();
    
    //-------------------------------------------------------------------------
    // Index Lifecycle
    //-------------------------------------------------------------------------
    
    /**
     * Create a new index.
     * @param name Unique index name
     * @param type Type of index (HNSW, DiskANN, etc.)
     * @param config Index configuration
     * @return Handle to the new index
     * @throws std::runtime_error if name exists or creation fails
     */
    IndexHandle create_index(const std::string& name, IndexType type,
                            const IndexConfig& config);
    
    /**
     * Create a new HNSW index (simplified API).
     * @param name Unique index name
     * @param dimensions Vector dimensions
     * @param max_elements Maximum number of elements
     * @param M HNSW M parameter (default: 16)
     * @param ef_construction HNSW ef_construction parameter (default: 200)
     * @return Handle to the new index
     */
    IndexHandle create_hnsw_index(const std::string& name,
                                   uint32_t dimensions,
                                   uint64_t max_elements,
                                   size_t M = 16,
                                   size_t ef_construction = 200);
    
    /**
     * Create a new DiskANN index (simplified API).
     * @param name Unique index name
     * @param dimensions Vector dimensions
     * @param max_elements Maximum number of elements
     * @param R_max_degree DiskANN R parameter (default: 64)
     * @param L_build DiskANN L parameter (default: 100)
     * @param alpha DiskANN alpha parameter (default: 1.2)
     * @return Handle to the new index
     */
    IndexHandle create_diskann_index(const std::string& name,
                                      uint32_t dimensions,
                                      uint64_t max_elements,
                                      uint32_t R_max_degree = 64,
                                      uint32_t L_build = 100,
                                      float alpha = 1.2f);
    
    /**
     * Create a new Text index (BM25).
     * @param name Unique index name
     * @param analyzer Analyzer type ("standard", "whitespace", "none")
     * @param language Language for stemming
     * @param k1 BM25 k1 parameter
     * @param b BM25 b parameter
     * @return Handle to the new index
     */
    IndexHandle create_text_index(const std::string& name,
                                   const std::string& analyzer = "standard",
                                   const std::string& language = "english",
                                   float k1 = 1.2f,
                                   float b = 0.75f);
    
    /**
     * Create a new BTree/Metadata index.
     * @param name Unique index name
     * @param fields Fields to index (up to 3 for composite)
     * @param unique Whether to enforce uniqueness
     * @return Handle to the new index
     */
    IndexHandle create_btree_index(const std::string& name,
                                    const std::vector<std::string>& fields,
                                    bool unique = false);
    
    /**
     * Open an existing index.
     * @param name Index name
     * @return Handle to the index
     * @throws std::runtime_error if index not found
     */
    IndexHandle open_index(const std::string& name);
    
    /**
     * Drop an index (delete permanently).
     * @param name Index name
     * @throws std::runtime_error if index not found or in use
     */
    void drop_index(const std::string& name);
    
    /**
     * Check if an index exists.
     */
    bool index_exists(const std::string& name) const;
    
    /**
     * List all active indexes.
     */
    std::vector<IndexInfo> list_indexes() const;
    
    /**
     * Get info for a specific index.
     */
    IndexInfo get_index_info(const std::string& name) const;
    
    /**
     * Get HNSW config for an index (only valid for HNSW type indices).
     */
    HNSWConfig get_hnsw_config(const std::string& name) const;
    
    /**
     * Get Text index config (only valid for TEXT type indices).
     */
    TextTypeMetadata get_text_config(const std::string& name) const;
    
    /**
     * Update Text index config (e.g., BTree slot ID, vocab size).
     */
    void update_text_config(const std::string& name, const TextTypeMetadata& config);
    
    /**
     * Get BTree index config (only valid for BTREE type indices).
     */
    BTreeTypeMetadata get_btree_config(const std::string& name) const;
    
    /**
     * Get Collection metadata (only valid for COLLECTION type indices).
     */
    CollectionTypeMetadata get_collection_config(const std::string& name) const;
    
    /**
     * Update Collection metadata (e.g., ID B-tree root PID, doc count).
     */
    void update_collection_config(const std::string& name, const CollectionTypeMetadata& config);

    //-------------------------------------------------------------------------
    // Internal Access (for IndexHandle)
    //-------------------------------------------------------------------------
    
    /**
     * Get the shared buffer manager.
     */
    BufferManager* buffer_manager() const { return buffer_manager_; }
    
    /**
     * Set the buffer manager for this catalog.
     * Must be called before initialize() to enable proper index registration.
     */
    void setBufferManager(BufferManager* bm_arg) { buffer_manager_ = bm_arg; }
    
    /**
     * Get the multi-file storage layer.
     */
    MultiFileStorage& storage() { return storage_; }
    const MultiFileStorage& storage() const { return storage_; }
    
    /**
     * Update element count for an index.
     */
    void update_index_element_count(uint32_t index_id, uint64_t count);
    
    /**
     * Update allocated pages count for an index.
     * This should be called before shutdown to persist the current allocation size.
     */
    void update_index_alloc_pages(uint32_t index_id, uint64_t alloc_pages);
    
    /**
     * Get the allocated pages count for an index.
     */
    uint64_t get_index_alloc_pages(uint32_t index_id) const;
    
    /**
     * Get the data directory.
     */
    const std::string& data_dir() const { return storage_.data_dir(); }
    
private:
    IndexCatalog() = default;
    ~IndexCatalog();
    
    // Catalog file operations
    void load_catalog();
    void save_catalog_header();
    void save_index_entry(const IndexEntry& entry);
    IndexEntry* find_entry_by_id(uint32_t index_id);
    IndexEntry* find_entry_by_name(const std::string& name);
    uint32_t allocate_index_id();
    
    // State
    std::atomic<bool> initialized_{false};
    mutable std::shared_mutex catalog_mutex_;
    int lock_fd_ = -1;  // File descriptor for directory lock
    
    // Storage
    MultiFileStorage storage_;
    BufferManager* buffer_manager_ = nullptr;
    
    // In-memory catalog cache
    CatalogHeader header_;
    std::vector<IndexEntry> entries_;
    std::unordered_map<std::string, uint32_t> name_to_index_id_;
};

} // namespace caliby
