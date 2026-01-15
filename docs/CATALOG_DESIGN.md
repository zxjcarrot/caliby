# Caliby Catalog System Design

## Overview

The Caliby Catalog System provides a unified storage layer for managing multiple vector indexes (HNSW, DiskANN, IVF) that share a common buffer pool while maintaining separate backing files. This design follows the **Multi-Level Calico** approach with translation path caching.

## Key Design Goals

1. **Multi-Index Support**: Multiple indexes with different names share the same buffer pool
2. **Separate Files**: Each index stored in its own file for isolation and management
3. **Translation Path Caching**: Amortize higher-level lookups across page accesses
4. **Unified Metadata**: Single catalog file tracks all indexes in the system
5. **Buffer Pool Integration**: All files (including catalog) managed through Calico buffer pool

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│                  (HNSW, DiskANN, IVF Indexes)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Index Catalog Manager                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ IndexHandle │  │ IndexHandle │  │ IndexHandle │   ...        │
│  │  (cached)   │  │  (cached)   │  │  (cached)   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Multi-Level Translation                         │
│                                                                   │
│   Global PageId = (index_id << 40) | local_page_id               │
│                                                                   │
│   Level 1: index_id → FileDescriptor + TranslationTable*         │
│   Level 2: local_page_id → frame_id (via cached table ptr)       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-File Storage Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ catalog.cal  │  │ idx_hnsw.cal │  │ idx_disk.cal │  ...      │
│  │  (index=0)   │  │  (index=1)   │  │  (index=2)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Calico Buffer Pool                             │
│           (Shared frames, unified eviction policy)                │
└─────────────────────────────────────────────────────────────────┘
```

## Page ID Structure

Following multi-level Calico, we split the 64-bit PageId:

```
┌────────────────────┬────────────────────────────────────────────┐
│     index_id       │              local_page_id                  │
│     (24 bits)      │                (40 bits)                    │
└────────────────────┴────────────────────────────────────────────┘
        │                           │
        │                           └─→ Page within index (up to 1TB with 4KB pages)
        └─────────────────────────────→ Index identifier (up to 16M indexes)
```

- **index_id = 0**: Reserved for the catalog file itself
- **index_id >= 1**: User-created indexes

## Translation Path Caching

The key optimization from Multi-Level Calico:

1. **First Access**: Thread looks up `index_id` → gets `(FileDescriptor, TranslationTable*)` 
2. **Cache Result**: Store pointer in thread-local or IndexHandle
3. **Subsequent Accesses**: Use cached pointer, index directly with `local_page_id`

```cpp
class IndexHandle {
    uint32_t index_id;
    int file_descriptor;           // Cached file descriptor
    TranslationEntry* table_ptr;   // Cached last-level table pointer
    
    PageId make_global_pid(uint64_t local_pid) {
        return (uint64_t(index_id) << 40) | local_pid;
    }
};
```

## Catalog File Format

The catalog file (`caliby_catalog`) uses the first index slot (index_id=0):

### Catalog Header (Page 0)
```cpp
struct CatalogHeader {
    uint64_t magic;           // 0xCAL1B7CA7A10G00  
    uint32_t version;         // Catalog format version
    uint32_t num_indexes;     // Number of active indexes
    uint32_t next_index_id;   // Next available index_id
    uint32_t flags;           // Reserved flags
    uint64_t checksum;        // Header checksum
    // Padding to page size
};
```

### Index Entry (Pages 1+)
```cpp
struct IndexEntry {
    uint32_t index_id;        // Unique index identifier
    uint32_t index_type;      // HNSW=1, DiskANN=2, IVF=3
    uint32_t status;          // ACTIVE=1, DELETED=2, CREATING=3
    uint32_t dimensions;      // Vector dimensions
    uint64_t max_elements;    // Maximum capacity
    uint64_t num_elements;    // Current element count
    uint64_t create_time;     // Creation timestamp
    uint64_t modify_time;     // Last modification timestamp
    char name[256];           // Index name (null-terminated)
    char file_path[512];      // Backing file path
    uint8_t type_metadata[256]; // Type-specific metadata
};
```

Multiple entries fit per page for efficiency.

## Index Types

```cpp
enum class IndexType : uint32_t {
    CATALOG = 0,    // Reserved for catalog
    HNSW = 1,
    DISKANN = 2,
    IVF = 3,
    // Future types...
};
```

## API Design

### IndexCatalog (Singleton)

```cpp
class IndexCatalog {
public:
    static IndexCatalog& instance();
    
    // Initialize catalog in directory
    void initialize(const std::string& data_dir);
    
    // Index lifecycle
    IndexHandle create_index(const std::string& name, IndexType type, 
                            const IndexConfig& config);
    IndexHandle open_index(const std::string& name);
    void drop_index(const std::string& name);
    
    // Enumeration
    std::vector<IndexInfo> list_indexes();
    
    // Buffer pool access (shared)
    BufferManager& buffer_manager();
    
private:
    std::string data_dir_;
    std::unordered_map<std::string, IndexEntry> name_to_entry_;
    std::unordered_map<uint32_t, FileHandle> index_files_;
    BufferManager* bm_;
};
```

### IndexHandle

```cpp
class IndexHandle {
public:
    // Page access with translation path caching
    template<typename PageType>
    PageType* pin_page(uint64_t local_page_id);
    
    void unpin_page(uint64_t local_page_id);
    
    template<typename Func>
    void optimistic_read(uint64_t local_page_id, Func&& read_func);
    
    // Allocate new page
    uint64_t allocate_page();
    
    // Metadata
    uint32_t index_id() const;
    const std::string& name() const;
    IndexType type() const;
    
private:
    uint32_t index_id_;
    int file_fd_;
    TranslationEntry* cached_table_;  // Translation path cache
    IndexCatalog* catalog_;
};
```

## Multi-File Storage

### File Naming Convention
```
<data_dir>/
├── caliby_catalog           # Catalog metadata (index_id=0)
├── caliby_<type>_<id>_<name>.dat    # Index data files (e.g., caliby_diskann_1_myindex.dat)
└── caliby_<type>_<id>_<name>.wal    # Optional WAL files
```

### File Handle Management

```cpp
class MultiFileStorage {
public:
    // Open/create file for index
    int open_index_file(uint32_t index_id, const std::string& path, bool create);
    
    // Close file
    void close_index_file(uint32_t index_id);
    
    // Read/write with global page ID
    void read_page(PageId global_pid, void* buffer);
    void write_page(PageId global_pid, const void* buffer);
    
private:
    std::unordered_map<uint32_t, int> index_to_fd_;
    std::shared_mutex fd_mutex_;
};
```

## Integration with HNSW/DiskANN

### Before (Single Heap File)
```cpp
class HNSW {
    BufferManager* bm_;  // Uses global buffer manager
    // Pages allocated from global page space
};
```

### After (Catalog-Managed)
```cpp
class HNSW {
    IndexHandle handle_;  // Obtained from catalog
    
    void addPoint(const float* vec, uint32_t& id) {
        // Use handle for page access - translation path cached
        auto* page = handle_.pin_page<HNSWNodePage>(node_page_id);
        // ...
        handle_.unpin_page(node_page_id);
    }
};
```

## Concurrency

1. **Catalog Modifications**: Protected by catalog-level mutex
2. **Index Creation/Deletion**: Atomic status transitions in IndexEntry
3. **Page Access**: Uses Calico's lock-free optimistic reads and CAS-based locking
4. **Translation Table**: Per-index tables, no cross-index contention

## Recovery

1. **Catalog Recovery**: On startup, scan catalog file, rebuild name→entry map
2. **Index Recovery**: Each index handles its own recovery using IndexHandle
3. **Crash Consistency**: 
   - Catalog updates use WAL or atomic page writes
   - Index creation: CREATING → flush → ACTIVE
   - Index deletion: ACTIVE → DELETED → remove file → remove entry

## Example Usage

```cpp
// Initialize catalog
auto& catalog = IndexCatalog::instance();
catalog.initialize("/data/caliby");

// Create HNSW index
IndexConfig config;
config.dimensions = 128;
config.max_elements = 1000000;
config.hnsw.M = 16;
config.hnsw.ef_construction = 200;

auto handle = catalog.create_index("my_vectors", IndexType::HNSW, config);

// Use index
HNSW index(handle);
index.addPoints(vectors);

// Later: open existing index
auto handle2 = catalog.open_index("my_vectors");
HNSW index2(handle2);
auto results = index2.searchKnn(query, k, ef);

// List all indexes
for (auto& info : catalog.list_indexes()) {
    std::cout << info.name << " (" << info.type << "): " 
              << info.num_elements << " vectors\n";
}

// Drop index
catalog.drop_index("my_vectors");
```

## Future Extensions

1. **Index Aliases**: Multiple names for same index
2. **Index Cloning**: Copy index with new name
3. **Tiered Storage**: Hot/cold index separation
4. **Distributed Catalog**: Multi-node index discovery
