# Caliby Developer Guide: Buffer Manager and Index Implementation

This guide explains how to implement new index types or data structures using Caliby's buffer manager. It covers page management, locking protocols, crash recovery, and index registration.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Page Guards: GuardO, GuardX, GuardS](#page-guards-guardo-guardx-guards)
3. [Allocating Pages](#allocating-pages)
4. [Optimistic Latch Coupling (OLC)](#optimistic-latch-coupling-olc)
5. [Implementing a New Index Type](#implementing-a-new-index-type)
6. [Index Registration and Recovery](#index-registration-and-recovery)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)

---

## Architecture Overview

Caliby uses a **buffer manager** that provides:
- **Page-based storage**: All data is stored in fixed-size pages (default 64KB)
- **Multi-index support**: Each index gets its own PID (Page ID) namespace
- **Optimistic concurrency**: High-performance locking using version numbers
- **Crash recovery**: Dirty pages are persisted; metadata enables recovery

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Bindings                          │
├─────────────────────────────────────────────────────────────┤
│  Collection  │   HNSW   │  DiskANN  │  IVF-PQ  │  B-Tree   │
├─────────────────────────────────────────────────────────────┤
│                     Index Catalog                            │
├─────────────────────────────────────────────────────────────┤
│                     Buffer Manager                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Page Guards │  │ PID Allocator│  │ Translation Array  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              Storage Backend (Files / io_uring)              │
└─────────────────────────────────────────────────────────────┘
```

### Global Buffer Manager

The buffer manager is a global singleton accessed via `bm`:

```cpp
#include "calico.hpp"

// Global buffer manager reference
extern BufferManager& bm;  // Defined in calico.cpp
```

---

## Page Guards: GuardO, GuardX, GuardS

Caliby provides three types of page guards that implement RAII-style page management:

### GuardO - Optimistic Read Guard

`GuardO` provides **optimistic read access** without acquiring any lock. It records a version number and validates it when the guard is released.

```cpp
#include "calico.hpp"

// Read a page optimistically
void read_page_example(PID page_id) {
    try {
        GuardO<MyPageType> guard(page_id);
        
        // Access page data (read-only)
        uint64_t value = guard->some_field;
        
        // Version is validated when guard goes out of scope
        // If another thread modified the page, OLCRestartException is thrown
        
    } catch (const OLCRestartException&) {
        // Page was modified - retry the operation
        return read_page_example(page_id);  // Simple retry
    }
}
```

**Key Properties:**
- No lock is acquired - highest concurrency
- Validation happens at destruction or explicit `checkVersionAndRestart()`
- Throws `OLCRestartException` if version changed
- Use for read-heavy workloads

### GuardX - Exclusive Write Guard

`GuardX` provides **exclusive write access** to a page. Only one thread can hold a GuardX on a page at a time.

```cpp
// Write to a page exclusively
void write_page_example(PID page_id) {
    GuardX<MyPageType> guard(page_id);
    
    // Page is now exclusively locked
    guard->some_field = 42;
    guard->dirty = true;  // Mark page as dirty for persistence
    
    // Lock is released when guard goes out of scope
}
```

**Upgrading from GuardO to GuardX:**

```cpp
void upgrade_example(PID page_id) {
    for (;;) {  // Retry loop
        try {
            GuardO<MyPageType> read_guard(page_id);
            
            // Read some data to decide if write is needed
            if (read_guard->needs_update) {
                // Upgrade to exclusive - may throw if version changed
                GuardX<MyPageType> write_guard(std::move(read_guard));
                
                // Now we have exclusive access
                write_guard->some_field = 42;
                write_guard->dirty = true;
                return;
            }
            return;  // No update needed
            
        } catch (const OLCRestartException&) {
            continue;  // Retry
        }
    }
}
```

### GuardS - Shared Read Guard

`GuardS` provides **shared read access** with a lock. Multiple threads can hold GuardS on the same page, but no GuardX can be acquired while GuardS is held.

```cpp
// Read with shared lock (blocks writers)
void shared_read_example(PID page_id) {
    GuardS<MyPageType> guard(page_id);
    
    // Shared lock held - other readers allowed, writers blocked
    uint64_t value = guard->some_field;
    
    // Lock released at scope exit
}
```

**Use GuardS when:**
- You need to hold a page for an extended operation
- You can't afford to restart due to concurrent modifications
- Multiple readers need guaranteed consistent view

### Guard Comparison

| Guard | Lock Type | Concurrency | Use Case |
|-------|-----------|-------------|----------|
| `GuardO` | Optimistic (none) | Highest | Short reads, can retry |
| `GuardS` | Shared | Medium | Long reads, no restarts |
| `GuardX` | Exclusive | Lowest | Writes |

---

## Allocating Pages

### Using AllocGuard

`AllocGuard` allocates a new page and provides exclusive access:

```cpp
// Allocate a new page with constructor parameters
void allocate_example(PIDAllocator* allocator) {
    // Allocate and construct with parameters
    AllocGuard<MyPageType> guard(allocator, arg1, arg2);
    
    // Get the page ID of the newly allocated page
    PID new_page_id = guard.pid;
    
    // Initialize page data
    guard->field1 = value1;
    guard->dirty = true;
    
    // Page is automatically marked dirty
}
```

### Using PIDAllocator

Each index should have its own `PIDAllocator` for page allocation:

```cpp
class MyIndex {
private:
    uint32_t index_id_;
    PIDAllocator* allocator_;
    
public:
    MyIndex(uint32_t index_id, uint64_t max_pages) 
        : index_id_(index_id) {
        // Get or create allocator for this index
        allocator_ = bm.getOrCreateAllocatorForIndex(index_id_, max_pages);
    }
    
    PID allocate_page() {
        // Allocate page in this index's namespace
        Page* page = bm.allocPageForIndex(index_id_, allocator_);
        return bm.toPID(page);
    }
};
```

### Page ID Encoding

In multi-index mode, page IDs encode the index ID:

```cpp
// Page ID structure (64-bit):
// [index_id: 32 bits][local_page_id: 32 bits]

PID encode_pid(uint32_t index_id, uint32_t local_page_id) {
    return (static_cast<PID>(index_id) << 32) | local_page_id;
}

uint32_t get_index_id(PID pid) {
    return static_cast<uint32_t>(pid >> 32);
}

uint32_t get_local_page_id(PID pid) {
    return static_cast<uint32_t>(pid & 0xFFFFFFFF);
}
```

---

## Optimistic Latch Coupling (OLC)

OLC is the concurrency protocol used by Caliby. It enables high concurrency by avoiding locks for read operations.

### Basic Pattern

```cpp
template<typename Func>
auto with_olc_retry(Func&& operation) {
    for (uint64_t retry_count = 0; ; retry_count++) {
        try {
            return operation();
        } catch (const OLCRestartException&) {
            // Exponential backoff on high contention
            if (retry_count > 100) {
                std::this_thread::yield();
            }
        }
    }
}

// Usage
auto result = with_olc_retry([&]() {
    GuardO<MyPage> guard(page_id);
    return guard->some_value;
});
```

### Tree Traversal with OLC

For tree structures, use parent-to-child latch coupling:

```cpp
void tree_lookup(uint64_t key) {
    for (;;) {
        try {
            // Start at root
            GuardO<TreeNode> current(root_pid);
            
            while (!current->is_leaf) {
                // Find child
                PID child_pid = current->find_child(key);
                
                // Move to child (validates current's version)
                GuardO<TreeNode> child(child_pid);
                current = std::move(child);
            }
            
            // Found leaf - extract value
            return current->lookup(key);
            
        } catch (const OLCRestartException&) {
            continue;  // Restart from root
        }
    }
}
```

### Upgrade Pattern for Modifications

```cpp
void tree_insert(uint64_t key, uint64_t value) {
    for (;;) {
        try {
            // Optimistic traversal to find leaf
            GuardO<TreeNode> parent(root_pid);
            GuardO<TreeNode> current = find_leaf(parent, key);
            
            // Check if split is needed
            if (current->needs_split()) {
                // Need exclusive access to parent and current
                GuardX<TreeNode> parent_x(std::move(parent));
                GuardX<TreeNode> current_x(std::move(current));
                
                // Perform split
                split_node(parent_x, current_x, key, value);
            } else {
                // Simple insert - upgrade current only
                GuardX<TreeNode> current_x(std::move(current));
                current_x->insert(key, value);
                current_x->dirty = true;
            }
            return;
            
        } catch (const OLCRestartException&) {
            continue;
        }
    }
}
```

---

## Implementing a New Index Type

### Step 1: Define Page Structure

```cpp
// include/caliby/my_index.hpp

#pragma once
#include "calico.hpp"

// Page structure must fit within pageSize (default 64KB)
struct MyIndexPage : public Page {
    static constexpr size_t HeaderSize = sizeof(Page) + 32;
    static constexpr size_t DataSize = pageSize - HeaderSize;
    
    // Header fields
    uint64_t entry_count;
    uint64_t next_page;
    uint32_t page_type;  // e.g., LEAF, INTERNAL
    uint32_t reserved;
    
    // Data area
    uint8_t data[DataSize];
    
    // Helper methods
    bool is_full() const { return entry_count >= max_entries(); }
    static constexpr size_t max_entries() { return DataSize / sizeof(Entry); }
};

static_assert(sizeof(MyIndexPage) == pageSize, "Page size mismatch");
```

### Step 2: Define Metadata Page

```cpp
struct MyIndexMetadata {
    static constexpr uint8_t VALID_MARKER = 0xAB;
    
    uint8_t valid;           // VALID_MARKER if initialized
    uint8_t reserved[3];
    uint32_t version;        // Schema version for compatibility
    
    // Index parameters (for recovery validation)
    uint64_t max_elements;
    uint32_t dimension;
    uint32_t param1;
    
    // Runtime state
    std::atomic<uint64_t> element_count;
    PID root_page;
    PID first_leaf;
    
    bool isValid() const { return valid == VALID_MARKER; }
    void markValid() { valid = VALID_MARKER; }
    void invalidate() { valid = 0; }
};
```

### Step 3: Implement Index Class

```cpp
// src/my_index.cpp

#include "my_index.hpp"
#include "catalog.hpp"
#include "logging.hpp"

class MyIndex {
private:
    uint32_t index_id_;
    std::string name_;
    PIDAllocator* allocator_;
    PID metadata_pid_;
    
    // Cached from metadata
    PID root_pid_;
    uint64_t max_elements_;
    
public:
    MyIndex(uint32_t index_id, const std::string& name,
            uint64_t max_elements, bool skip_recovery = false)
        : index_id_(index_id)
        , name_(name)
        , max_elements_(max_elements) 
    {
        // Calculate pages needed
        uint64_t data_pages = (max_elements + entries_per_page - 1) / entries_per_page;
        uint64_t total_pages = 1 + data_pages + (data_pages / 10);  // +10% buffer
        
        // Get allocator
        allocator_ = bm.getOrCreateAllocatorForIndex(index_id_, total_pages);
        
        // Compute metadata page ID
        if (bm.supportsMultiIndexPIDs() && index_id_ > 0) {
            metadata_pid_ = (static_cast<PID>(index_id_) << 32) | 0ULL;
        } else {
            metadata_pid_ = 0;
        }
        
        // Initialize or recover
        initialize_or_recover(skip_recovery);
    }
    
private:
    void initialize_or_recover(bool skip_recovery) {
        GuardX<MetaDataPage> meta_guard(metadata_pid_);
        MyIndexMetadata* meta = get_my_metadata(meta_guard.ptr);
        
        bool can_recover = !skip_recovery && meta->isValid() &&
                           meta->max_elements == max_elements_;
        
        if (can_recover) {
            // Recovery path
            CALIBY_LOG_INFO("MyIndex", "Recovering index ", name_);
            root_pid_ = meta->root_page;
            // ... restore other state
        } else {
            // Fresh initialization
            CALIBY_LOG_INFO("MyIndex", "Creating new index ", name_);
            
            // Invalidate old metadata if exists
            if (meta->isValid()) {
                meta->invalidate();
            }
            
            // Allocate root page
            AllocGuard<MyIndexPage> root_guard(allocator_);
            root_pid_ = root_guard.pid;
            root_guard->entry_count = 0;
            root_guard->page_type = PAGE_TYPE_ROOT;
            
            // Update metadata
            meta->max_elements = max_elements_;
            meta->root_page = root_pid_;
            meta->element_count.store(0);
            meta->markValid();
            meta_guard->dirty = true;
        }
    }
};
```

### Step 4: Register with Catalog

```cpp
// In your index constructor or factory function

#include "catalog.hpp"

MyIndex* create_my_index(const std::string& name, uint64_t max_elements) {
    // Get unique index ID from catalog
    auto& catalog = caliby::IndexCatalog::instance();
    
    // Create index entry
    caliby::IndexHandle handle = catalog.create_index(
        name,
        caliby::IndexType::CUSTOM,  // Or add new type
        0,      // dimensions (0 if not applicable)
        max_elements
    );
    
    uint32_t index_id = handle.index_id();
    CALIBY_LOG_INFO("MyIndex", "Registered with catalog: id=", index_id, " name=", name);
    
    return new MyIndex(index_id, name, max_elements);
}
```

---

## Index Registration and Recovery

### Index Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Create    │───▶│   Active    │───▶│   Close     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Register in │    │ Operations  │    │ Flush dirty │
│  Catalog    │    │ (R/W/Query) │    │   pages     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Allocate    │    │ Mark pages  │    │ Persist     │
│ PID space   │    │   dirty     │    │  metadata   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Recovery Process

When the system restarts, recovery happens automatically:

```cpp
void recover_index(uint32_t index_id, const std::string& name) {
    // 1. Catalog provides index metadata
    auto& catalog = caliby::IndexCatalog::instance();
    auto info = catalog.get_index_info(index_id);
    
    // 2. Open index with same parameters (triggers recovery)
    MyIndex index(
        index_id,
        name,
        info.max_elements,
        false  // skip_recovery = false
    );
    
    // 3. Index constructor reads metadata page
    // 4. If metadata valid and params match, restore state
    // 5. Otherwise, reinitialize
}
```

### Metadata Page Pattern

Every index should use page 0 (local) for metadata:

```cpp
// Metadata is always at local page 0 for the index
PID get_metadata_pid(uint32_t index_id) {
    if (bm.supportsMultiIndexPIDs() && index_id > 0) {
        return (static_cast<PID>(index_id) << 32) | 0ULL;
    }
    return 0;  // Single-index mode
}

// Recovery check pattern
bool can_recover(GuardX<MetaDataPage>& meta_guard, const Params& params) {
    auto* meta = get_index_metadata(meta_guard.ptr);
    
    return meta->isValid() &&
           meta->version == CURRENT_VERSION &&
           meta->param1 == params.param1 &&
           meta->param2 == params.param2;
}
```

---

## Best Practices

### 1. Always Mark Dirty Pages

```cpp
void update_page(PID pid) {
    GuardX<MyPage> guard(pid);
    guard->data = new_value;
    guard->dirty = true;  // CRITICAL: Mark dirty for persistence
}
```

### 2. Use Appropriate Guard Types

```cpp
// READ: Use GuardO for short reads
uint64_t quick_read(PID pid) {
    GuardO<MyPage> guard(pid);
    return guard->value;  // May throw OLCRestartException
}

// READ: Use GuardS for long reads or when restart is expensive
void long_read(PID pid, std::vector<uint64_t>& results) {
    GuardS<MyPage> guard(pid);
    // Copy all data - no risk of restart
    for (size_t i = 0; i < guard->count; i++) {
        results.push_back(guard->entries[i]);
    }
}

// WRITE: Always use GuardX
void write_data(PID pid, uint64_t value) {
    GuardX<MyPage> guard(pid);
    guard->value = value;
    guard->dirty = true;
}
```

### 3. Handle OLCRestartException

```cpp
// Always wrap GuardO operations in try-catch with retry
void safe_read(PID pid) {
    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        try {
            GuardO<MyPage> guard(pid);
            // ... read operations
            return;
        } catch (const OLCRestartException&) {
            // Optionally log on high retry count
            if (retry > 10) {
                CALIBY_LOG_WARN("MyIndex", "High contention on page ", pid);
            }
        }
    }
    throw std::runtime_error("Max retries exceeded");
}
```

### 4. Minimize Lock Hold Time

```cpp
// BAD: Holding exclusive lock while doing I/O
void bad_pattern(PID pid) {
    GuardX<MyPage> guard(pid);
    expensive_computation();  // DON'T DO THIS
    network_call();           // DON'T DO THIS
}

// GOOD: Prepare data, then acquire lock briefly
void good_pattern(PID pid, const Data& data) {
    // Prepare data outside lock
    auto prepared = prepare(data);
    
    // Brief exclusive access
    GuardX<MyPage> guard(pid);
    guard->copy_from(prepared);
    guard->dirty = true;
}
```

### 5. Use Logging Appropriately

```cpp
// DEBUG: Verbose internal details
CALIBY_LOG_DEBUG("MyIndex", "Traversing node ", node_id, " level=", level);

// INFO: Important operations
CALIBY_LOG_INFO("MyIndex", "Index created: ", name_, " max_elements=", max_elements_);

// WARN: Recoverable issues
CALIBY_LOG_WARN("MyIndex", "Page split required, high fragmentation");

// ERROR: Serious problems
CALIBY_LOG_ERROR("MyIndex", "Failed to allocate page: out of space");
```

---

## Common Pitfalls

### 1. Forgetting to Mark Dirty

```cpp
// BUG: Page won't be persisted!
void buggy_update(PID pid) {
    GuardX<MyPage> guard(pid);
    guard->value = 42;
    // MISSING: guard->dirty = true;
}
```

### 2. Deadlock from Guard Order

```cpp
// POTENTIAL DEADLOCK: Inconsistent lock ordering
void thread1() {
    GuardX<MyPage> a(pid_a);
    GuardX<MyPage> b(pid_b);  // Waits for b
}

void thread2() {
    GuardX<MyPage> b(pid_b);
    GuardX<MyPage> a(pid_a);  // Waits for a -> DEADLOCK
}

// SOLUTION: Always acquire locks in consistent order (e.g., by PID)
void safe_multi_lock(PID pid1, PID pid2) {
    if (pid1 < pid2) {
        GuardX<MyPage> g1(pid1);
        GuardX<MyPage> g2(pid2);
    } else {
        GuardX<MyPage> g2(pid2);
        GuardX<MyPage> g1(pid1);
    }
}
```

### 3. Holding Guards Across Allocations

```cpp
// BUG: Allocation might trigger eviction, invalidating guards
void buggy_alloc(PID existing_pid) {
    GuardO<MyPage> guard(existing_pid);
    
    // DANGER: This allocation might evict the page we're reading!
    AllocGuard<MyPage> new_page(allocator_);
    
    // guard might now be invalid!
    auto value = guard->field;  // UNDEFINED BEHAVIOR
}

// SOLUTION: Release guard before allocation
void safe_alloc(PID existing_pid) {
    uint64_t value;
    {
        GuardO<MyPage> guard(existing_pid);
        value = guard->field;
    }  // Guard released
    
    AllocGuard<MyPage> new_page(allocator_);
    // Safe to use value here
}
```

### 4. Not Handling Recovery Parameters

```cpp
// BUG: Index with different parameters might corrupt data
MyIndex::MyIndex(uint32_t index_id, uint64_t max_elements) {
    GuardX<MetaDataPage> meta_guard(metadata_pid_);
    auto* meta = get_metadata(meta_guard.ptr);
    
    if (meta->isValid()) {
        // DANGER: Not checking if parameters match!
        root_pid_ = meta->root_page;  // Might be incompatible
    }
}

// SOLUTION: Always validate parameters
MyIndex::MyIndex(uint32_t index_id, uint64_t max_elements) {
    GuardX<MetaDataPage> meta_guard(metadata_pid_);
    auto* meta = get_metadata(meta_guard.ptr);
    
    bool can_recover = meta->isValid() &&
                       meta->max_elements == max_elements &&  // Check params!
                       meta->dimension == dimension_;
    
    if (can_recover) {
        root_pid_ = meta->root_page;
    } else {
        // Reinitialize
        meta->invalidate();
        // ... allocate fresh pages
    }
}
```

---

## Example: Complete Mini-Index

Here's a complete example of a simple key-value index:

```cpp
// mini_kv_index.hpp
#pragma once
#include "calico.hpp"
#include "logging.hpp"

struct KVEntry {
    uint64_t key;
    uint64_t value;
};

struct KVPage : public Page {
    static constexpr size_t MaxEntries = (pageSize - sizeof(Page) - 16) / sizeof(KVEntry);
    
    uint64_t entry_count;
    uint64_t next_page;
    KVEntry entries[MaxEntries];
};

class MiniKVIndex {
    uint32_t index_id_;
    PIDAllocator* allocator_;
    PID head_pid_;
    
public:
    MiniKVIndex(uint32_t index_id) : index_id_(index_id) {
        allocator_ = bm.getOrCreateAllocatorForIndex(index_id_, 1000);
        
        PID meta_pid = (static_cast<PID>(index_id_) << 32) | 0ULL;
        GuardX<MetaDataPage> meta(meta_pid);
        
        // Simple: always create fresh
        AllocGuard<KVPage> head(allocator_);
        head->entry_count = 0;
        head->next_page = 0;
        head_pid_ = head.pid;
        
        meta->dirty = true;
        CALIBY_LOG_INFO("MiniKV", "Created index ", index_id_);
    }
    
    void put(uint64_t key, uint64_t value) {
        for (;;) {
            try {
                GuardO<KVPage> page(head_pid_);
                if (page->entry_count < KVPage::MaxEntries) {
                    GuardX<KVPage> page_x(std::move(page));
                    auto idx = page_x->entry_count++;
                    page_x->entries[idx] = {key, value};
                    page_x->dirty = true;
                    return;
                }
                // TODO: Handle full page (allocate new)
                throw std::runtime_error("Page full");
            } catch (const OLCRestartException&) {
                continue;
            }
        }
    }
    
    std::optional<uint64_t> get(uint64_t key) {
        for (;;) {
            try {
                GuardO<KVPage> page(head_pid_);
                for (size_t i = 0; i < page->entry_count; i++) {
                    if (page->entries[i].key == key) {
                        return page->entries[i].value;
                    }
                }
                return std::nullopt;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
    }
};
```

---

## Further Reading

- [docs/COLLECTION_DESIGN.md](COLLECTION_DESIGN.md) - Collection implementation details
- [docs/CATALOG_DESIGN.md](CATALOG_DESIGN.md) - Index catalog system
- [include/caliby/calico.hpp](../include/caliby/calico.hpp) - Buffer manager API
- [src/hnsw.cpp](../src/hnsw.cpp) - HNSW implementation (real-world example)
- [src/diskann.cpp](../src/diskann.cpp) - DiskANN implementation (graph-based index)
