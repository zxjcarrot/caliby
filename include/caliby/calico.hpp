#ifndef CALICO_HPP
#define CALICO_HPP

#include <libaio.h>
#ifdef CALIBY_HAS_URING
#include <liburing.h>
#endif
#include <nmmintrin.h>  // For CRC32 and prefetch
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <span>
#include "hash_tables.hpp"

// Extern declarations must come BEFORE headers that use them
extern __thread uint16_t workerThreadId;
extern __thread int32_t tpcchistorycounter;

#include "exmap.h"
#include "tpcc/TPCCWorkload.hpp"
#include "tpcc/ZipfianGenerator.hpp"

using namespace tpcc;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u64 PID;  // page id type

static constexpr u64 pageSize = 4096 * 4;

__attribute__((target("sse4.2"))) uint64_t hash64(uint64_t key, uint32_t seed);

template <typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
};

struct alignas(pageSize) Page {
    bool dirty;
};

struct GraphNodePage {
    bool dirty;
    u32 neighborCount;
    u32 padding;
    u64 value;
    PID neighbors[1];

    static constexpr size_t headerSize() { return offsetof(GraphNodePage, neighbors); }
    static constexpr u32 maxNeighbors() { return static_cast<u32>((pageSize - headerSize()) / sizeof(PID)); }
    PID* neighborData() { return neighbors; }
    const PID* neighborData() const { return neighbors; }
};

static const int16_t maxWorkerThreads = 128;

#define die(msg)            \
    do {                    \
        perror(msg);        \
        exit(EXIT_FAILURE); \
    } while (0)

uint64_t rdtsc();

void* allocHuge(size_t size);

void yield(u64 counter);

struct PageState {
    std::atomic<u64> stateAndVersion;

    static const u64 Evicted = 0;
    static const u64 Unlocked = 1;
    static const u64 SharedBase = 2;
    static const u64 MaxShared = 253;
    static const u64 Locked = 254;
    static const u64 Marked = 255;

    static constexpr u64 stateMask = 0xFF00000000000000ULL;
    static constexpr u64 lowMask = 0x00FFFFFFFFFFFFFFULL;
    static constexpr u64 frameMask = 0x00000000FFFFFFFFULL;
    static constexpr u64 versionMaskPacked = 0x00FFFFFF00000000ULL;
    static constexpr u32 versionBitsPacked = 24;
    static constexpr u64 versionMaxPacked = (1ULL << versionBitsPacked) - 1;
    static constexpr u32 packedInvalidFrame = 0;  // Frame 0 is reserved and never allocated

    static bool packedMode;
    static void setPackedMode(bool value) { packedMode = value; }
    static bool isPackedMode() { return packedMode; }

    PageState() {}

    static inline u64 extractLow(u64 v) { return v & lowMask; }
    static inline u64 extractFrame(u64 v) { return v & frameMask; }

    static inline u64 compose(u64 state, u64 low) { return (state << 56) | (low & lowMask); }

    static inline u64 incrementLow(u64 oldStateAndVersion) {
        if (!packedMode) return (extractLow(oldStateAndVersion) + 1) & lowMask;
        u64 framePart = oldStateAndVersion & frameMask;
        u64 versionPart = (oldStateAndVersion & versionMaskPacked) >> 32;
        versionPart = (versionPart + 1) & versionMaxPacked;
        return ((versionPart << 32) & versionMaskPacked) | framePart;
    }

    void init() {
        // Zero-value entry design: all-zero represents evicted state
        stateAndVersion.store(0, std::memory_order_release);
    }

    static inline u64 sameVersion(u64 oldStateAndVersion, u64 newState) {
        return compose(newState, extractLow(oldStateAndVersion));
    }

    static inline u64 nextVersion(u64 oldStateAndVersion, u64 newState) {
        return compose(newState, incrementLow(oldStateAndVersion));
    }

    bool tryLockX(u64 oldStateAndVersion) {
        return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Locked));
    }

    void unlockX() {
        assert(getState() == Locked);
        stateAndVersion.store(nextVersion(stateAndVersion.load(), Unlocked), std::memory_order_release);
    }

    void unlockXEvicted() {
        assert(getState() == Locked);
        // Write zero for evicted state (zero-value entry design)
        stateAndVersion.store(0, std::memory_order_release);
    }

    void downgradeLock() {
        assert(getState() == Locked);
        stateAndVersion.store(nextVersion(stateAndVersion.load(), 1), std::memory_order_release);
    }

    bool tryLockS(u64 oldStateAndVersion) {
        u64 s = getState(oldStateAndVersion);
        if (s >= Unlocked && s < MaxShared)
            return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, s + 1));
        if (s == Marked)
            return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, SharedBase));
        return false;
    }

    void unlockS() {
        while (true) {
            u64 oldStateAndVersion = stateAndVersion.load();
            u64 state = getState(oldStateAndVersion);
            assert(state > Unlocked && state <= MaxShared);
            if (stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, state - 1)))
                return;
        }
    }

    bool tryMark(u64 oldStateAndVersion) {
        assert(getState(oldStateAndVersion) == Unlocked);
        return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Marked));
    }

    static u64 getState(u64 v) { return v >> 56; };
    u64 getState() { return getState(stateAndVersion.load()); }

    static u64 getFrameValue(u64 v) {
        if (!packedMode) return 0;
        return v & frameMask;
    }

    u64 getFrameValue() const {
        if (!packedMode) return 0;
        return extractFrame(stateAndVersion.load(std::memory_order_relaxed));
    }

    void setFrameValue(u64 frame) {
        if (!packedMode) return;
        frame &= frameMask;
        u64 expected = stateAndVersion.load(std::memory_order_relaxed);
        assert(getState(expected) == Locked);
        u64 desired = (expected & ~frameMask) | frame;
        stateAndVersion.store(desired, std::memory_order_release);
        return;
    }

    void clearFrameValue() {
        if (packedMode) setFrameValue(packedInvalidFrame);
    }

    void operator=(PageState&) = delete;

    static u64 extractVersion(u64 v) {
        if (!packedMode) return 0;
        return (v & versionMaskPacked) >> 32;
    }
};

struct ResidentPageSet {
    static const u64 empty = ~0ull;
    static const u64 tombstone = (~0ull) - 1;

    struct Entry {
        std::atomic<u64> pid;
    };

    Entry* ht;
    u64 count;
    u64 mask;
    std::atomic<u64> clockPos;

    ResidentPageSet(u64 maxCount);
    ~ResidentPageSet();

    u64 next_pow2(u64 x);
    u64 hash(u64 k);
    void insert(u64 pid);
    bool remove(u64 pid);
    
    // Remove all pages belonging to indexes with ID >= minIndexId
    void removeAllForIndexes(u32 minIndexId) {
        for (u64 i = 0; i < count; ++i) {
            u64 pid = ht[i].pid.load();
            if (pid != empty && pid != tombstone) {
                // Extract index ID from PID (top 32 bits)
                u32 indexId = static_cast<u32>(pid >> 32);
                if (indexId >= minIndexId) {
                    // Try to remove it
                    u64 expected = pid;
                    ht[i].pid.compare_exchange_strong(expected, tombstone);
                }
            }
        }
    }

    template <class Fn>
    void iterateClockBatch(u64 batch, Fn fn) {
        u64 pos, newPos;
        do {
            pos = clockPos.load();
            newPos = (pos + batch) % count;
        } while (!clockPos.compare_exchange_strong(pos, newPos));

        for (u64 i = 0; i < batch; i++) {
            u64 curr = ht[pos].pid.load();
            if ((curr != tombstone) && (curr != empty)) fn(curr);
            pos = (pos + 1) & mask;
        }
    }
};

struct BufferManager;

struct IOInterface {
    virtual void writePages(const std::vector<PID>& pages) = 0;
    virtual void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations) = 0;
    virtual ~IOInterface() {}
};

struct LibaioInterface: public IOInterface {
    static const u64 maxIOs = 256;

    int blockfd;
    BufferManager* bm_ptr;
    io_context_t ctx;
    iocb cb[maxIOs];
    iocb* cbPtr[maxIOs];
    io_event events[maxIOs];

    LibaioInterface(int blockfd, BufferManager* bm_ptr);
    ~LibaioInterface() { io_destroy(ctx); }
    virtual void writePages(const std::vector<PID>& pages);
    virtual void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations);
};

#ifdef CALIBY_HAS_URING
struct IOUringInterface: public IOInterface {
    static const u64 maxIOs = 256;
    static const u64 queueDepth = 512;

    int blockfd;
    BufferManager* bm_ptr;
    struct io_uring ring;

    IOUringInterface(int blockfd, BufferManager* bm_ptr);
    ~IOUringInterface();
    virtual void writePages(const std::vector<PID>& pages);
    virtual void readPages(const std::vector<PID>& pages, const std::vector<Page*>& destinations);
};
#endif

struct FreePartition {
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    std::vector<u32> frames;
    u32 top = 0;
    char cachline_pad0[64 - sizeof(std::atomic_flag) - sizeof(std::vector<u32>) - sizeof(u32)];
};

//=============================================================================
// Per-Index Translation Array with Hole-Punching Support
//=============================================================================

/**
 * Per-index translation array supporting atomic hole-punching.
 * Each index gets its own translation array that can be lazily allocated.
 * Supports up to 2^32 pages per index.
 */
struct IndexTranslationArray {
    // Use 4KB OS pages for fine-grained hole punching (not huge pages)
    static constexpr u64 TRANSLATION_OS_PAGE_SIZE = 4096;
    static constexpr u64 ENTRIES_PER_OS_PAGE = TRANSLATION_OS_PAGE_SIZE / sizeof(PageState);
    static constexpr u32 REF_COUNT_LOCK_BIT = 0x80000000u;
    static constexpr u32 REF_COUNT_MASK = 0x7FFFFFFFu;
    static constexpr u64 GROWTH_FACTOR = 2;  // Double capacity when growing
    static constexpr u64 MIN_INITIAL_CAPACITY = 1024;  // Minimum initial pages
    
    PageState* pageStates;              // Page state array for this index (mmap'd)
    std::atomic<u64> capacity;          // Allocated capacity (number of PageState entries) - atomic for lock-free reads
    std::atomic<u32>* refCounts;        // Reference counts per OS page group
    u64 numRefCountGroups;              // Number of ref count groups
    std::atomic<u64> allocCount;        // Per-index allocation counter for local PIDs
    mutable std::mutex growMutex;       // Mutex for thread-safe growth
    
    // Cached information for translation path caching
    int file_fd;                         // File descriptor for this index's data file
    u32 index_id;                        // Index ID for quick lookup
    
    IndexTranslationArray(u32 indexId, u64 maxPages, u64 initialAllocCount = 0, int fd = -1);
    ~IndexTranslationArray();
    
    // Disable copy
    IndexTranslationArray(const IndexTranslationArray&) = delete;
    IndexTranslationArray& operator=(const IndexTranslationArray&) = delete;
    
    inline PageState& get(u64 localPageId) {
        return pageStates[localPageId];
    }
    
    /**
     * Grow the translation array to accommodate at least minCapacity pages.
     * Uses mremap() for efficient in-place growth on Linux.
     * Thread-safe: multiple threads can call this concurrently.
     * @param minCapacity Minimum required capacity
     * @return true if growth succeeded or capacity already sufficient, false on failure
     */
    bool ensureCapacity(u64 minCapacity);
    
    // Hole punching support - increment ref count when page becomes resident
    void incrementRefCount(u64 localPageId);
    // Decrement ref count when page is evicted; may trigger hole punch
    void decrementRefCount(u64 localPageId, std::atomic<u64>& holePunchCounter);
    
    // Get OS page group index for a local page ID
    u64 getRefCountGroup(u64 localPageId) const {
        return (localPageId * sizeof(PageState)) / TRANSLATION_OS_PAGE_SIZE;
    }
};

//=============================================================================
// Two-Level Translation Array with Multi-Index Support
//=============================================================================

/**
 * Two-level page state array for multi-index support.
 * 
 * PID Layout: [index_id (32 bits)][local_page_id (32 bits)]
 * - Top level: indexed by index_id, pointers to per-index arrays
 * - Bottom level: per-index PageState arrays with hole-punching
 * 
 * Translation path caching: The get() method caches the last accessed
 * IndexTranslationArray pointer in thread-local storage to amortize
 * the top-level lookup cost for repeated accesses to the same index.
 */
struct TwoLevelPageStateArray {
    // PID bit layout: top 32 bits = index_id, bottom 32 bits = local_page_id
    static constexpr u32 INDEX_ID_BITS = 32;
    static constexpr u32 LOCAL_PAGE_BITS = 32;
    static constexpr u64 LOCAL_PAGE_MASK = 0xFFFFFFFFULL;
    static constexpr u32 MAX_INDEXES = 65536;  // Practical limit for top-level array
    
    // Top-level array: pointers to per-index translation arrays
    std::atomic<IndexTranslationArray*>* indexArrays;
    u32 numIndexSlots;                    // Actual number of slots allocated
    mutable std::shared_mutex indexMutex; // For registering/unregistering indexes
    
    // Generation counter - incremented when indexes are unregistered
    // Used to invalidate thread-local caches
    std::atomic<u64> generation{0};
    
    // Reference to BufferManager's hole punch counter
    std::atomic<u64>* holePunchCounterPtr;
    
    // Default array for index 0 (legacy single-index mode)
    // This ensures backward compatibility with code that doesn't use catalog
    IndexTranslationArray* defaultArray;
    
    TwoLevelPageStateArray(u64 defaultIndexCapacity = 0);
    ~TwoLevelPageStateArray();
    
    // Disable copy
    TwoLevelPageStateArray(const TwoLevelPageStateArray&) = delete;
    TwoLevelPageStateArray& operator=(const TwoLevelPageStateArray&) = delete;
    
    /**
     * Register a new index with given capacity.
     * @param indexId The index ID (must be < MAX_INDEXES)
     * @param maxPages Maximum number of pages for this index
     * @param initialAllocCount Initial allocation count from catalog
     * @param fileFd File descriptor for this index's data file (for caching)
     */
    void registerIndex(u32 indexId, u64 maxPages, u64 initialAllocCount = 0, int fileFd = -1);
    
    /**
     * Unregister an index and free its translation array.
     */
    void unregisterIndex(u32 indexId);
    
    /**
     * Unregister all non-zero indexes (for reset/close).
     * Index 0 is preserved as the default single-index mode.
     */
    void unregisterAllNonZero();
    
    /**
     * Check if an index is registered.
     */
    bool isIndexRegistered(u32 indexId) const;
    
    /**
     * Get the per-index translation array (for translation path caching).
     */
    IndexTranslationArray* getIndexArray(u32 indexId) const;
    
    /**
     * Main access method with translation path caching.
     * Extracts index_id and local_page_id from PID and returns PageState reference.
     */
    inline PageState& get(PID pid) {
        u32 indexId = static_cast<u32>(pid >> LOCAL_PAGE_BITS);
        u32 localPageId = static_cast<u32>(pid & LOCAL_PAGE_MASK);
        
        // Fast path: use thread-local cache for repeated access to same index
        // The cache stores the last accessed IndexTranslationArray pointer
        // Also cache generation to detect invalidation
        thread_local u32 cachedIndexId = 0xFFFFFFFF;
        thread_local IndexTranslationArray* cachedArray = nullptr;
        thread_local u64 cachedGeneration = 0;
        
        u64 currentGen = generation.load(std::memory_order_acquire);
        
        if (cachedIndexId == indexId && cachedGeneration == currentGen && cachedArray != nullptr) {
            // Cache hit - use cached array pointer
            return cachedArray->get(localPageId);
        }
        IndexTranslationArray* arr;

        // Cache miss - lookup in top-level array
        if (indexId == 0 && defaultArray != nullptr) {
            arr = defaultArray;
        } else {
            arr = indexArrays[indexId];
        }
        // Update cache
        cachedIndexId = indexId;
        cachedArray = arr;
        cachedGeneration = currentGen;
        
        return arr->get(localPageId);
    }

    /**
     * Optimized access method that bypasses TLS cache when IndexTranslationArray is already known.
     * Useful for tight loops where the same index is accessed repeatedly.
     */
    inline PageState& get(PID pid, IndexTranslationArray* arr) {
        u32 localPageId = static_cast<u32>(pid & LOCAL_PAGE_MASK);
        return arr->get(localPageId);
    }

    // Extract index_id from PID
    static inline u32 getIndexId(PID pid) {
        return static_cast<u32>(pid >> LOCAL_PAGE_BITS);
    }
    
    // Extract local_page_id from PID
    static inline u32 getLocalPageId(PID pid) {
        return static_cast<u32>(pid & LOCAL_PAGE_MASK);
    }
    
    // Compose PID from index_id and local_page_id
    static inline PID makePID(u32 indexId, u32 localPageId) {
        return (static_cast<u64>(indexId) << LOCAL_PAGE_BITS) | localPageId;
    }
    
    // Set the hole punch counter reference
    void setHolePunchCounter(std::atomic<u64>* counter) {
        holePunchCounterPtr = counter;
    }
    
    // Hole punching support - delegates to per-index array
    void incrementRefCount(PID pid);
    void decrementRefCount(PID pid);
    
    /**
     * Get the current capacity (in pages) of an index's translation array.
     * Returns 0 if index is not registered.
     */
    u64 getIndexCapacity(u32 indexId) const;
    
    /**
     * Get all registered index IDs with their capacities.
     * Used for persisting allocation state on shutdown.
     */
    std::vector<std::pair<u32, u64>> getAllIndexCapacities() const;
};

// Three-level indirection for PageState array
struct ThreeLevelPageStateArray {
    static constexpr u32 bottomBits = 19;   // Fixed at 19 bits for bottom-level index
    static constexpr u32 middleBits = 2;   // Fixed at 2 bits for middle-level index
    
    PageState*** top_level_arrays;    // Top-level array of pointers to middle-level arrays
    u32 num_top_slots;                // Number of top-level slots (2^topBits)
    u32 num_middle_slots;             // Number of middle-level slots per top slot (2^middleBits)
    u32 entries_per_slot;             // Number of entries per bottom-level array (2^bottomBits)
    u32 topBits;                      // Number of bits for top-level index
    u64 total_entries;                // Total number of entries
    
    ThreeLevelPageStateArray(u64 virtCount);
    ~ThreeLevelPageStateArray();
    
    inline PageState& get(PID pid) {
        u32 top_idx = static_cast<u32>(pid >> (bottomBits + middleBits));
        u32 middle_idx = static_cast<u32>((pid >> bottomBits) & ((1ULL << middleBits) - 1));
        u32 bottom_idx = static_cast<u32>(pid & ((1ULL << bottomBits) - 1));
        return top_level_arrays[top_idx][middle_idx][bottom_idx];
    }
};

struct PIDAllocator {
    std::atomic<u64> next{0};
    std::atomic<u64> end{0};
    std::mutex lock;
    u32 index_id;  // Which index this allocator belongs to
    
    PIDAllocator() : index_id(0) {}
    explicit PIDAllocator(u32 idx) : index_id(idx) {}
};

/**
 * Index Catalog - tracks per-index allocation counters
 * Persisted to disk for recovery
 */
struct IndexCatalog {
    struct IndexEntry {
        u32 index_id;
        std::atomic<u64> alloc_count;  // Next local PID to allocate
        u64 max_pages;                 // Capacity for this index
        int file_fd;                   // File descriptor (cached)
        
        IndexEntry() : index_id(0), alloc_count(1), max_pages(0), file_fd(-1) {}
        IndexEntry(u32 id, u64 max) : index_id(id), alloc_count(1), max_pages(max), file_fd(-1) {}
        
        // Custom copy constructor (loads atomic value)
        IndexEntry(const IndexEntry& other) 
            : index_id(other.index_id), 
              alloc_count(other.alloc_count.load(std::memory_order_relaxed)),
              max_pages(other.max_pages),
              file_fd(other.file_fd) {}
        
        // Custom move constructor (loads atomic value)
        IndexEntry(IndexEntry&& other) noexcept
            : index_id(other.index_id),
              alloc_count(other.alloc_count.load(std::memory_order_relaxed)),
              max_pages(other.max_pages),
              file_fd(other.file_fd) {}
        
        // Custom copy assignment
        IndexEntry& operator=(const IndexEntry& other) {
            if (this != &other) {
                index_id = other.index_id;
                alloc_count.store(other.alloc_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
                max_pages = other.max_pages;
                file_fd = other.file_fd;
            }
            return *this;
        }
        
        // Custom move assignment
        IndexEntry& operator=(IndexEntry&& other) noexcept {
            if (this != &other) {
                index_id = other.index_id;
                alloc_count.store(other.alloc_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
                max_pages = other.max_pages;
                file_fd = other.file_fd;
            }
            return *this;
        }
    };
    
    std::unordered_map<u32, IndexEntry> entries;
    mutable std::shared_mutex mutex;
    std::string catalog_file_path;
    
    IndexCatalog(const std::string& path = "./catalog.dat") : catalog_file_path(path) {}
    
    // Get or create an allocator for an index
    PIDAllocator* getOrCreateAllocator(u32 index_id, u64 max_pages = 0);
    
    // Update alloc_count for an index
    void updateAllocCount(u32 index_id, u64 count);
    
    // Get current alloc_count for an index
    u64 getAllocCount(u32 index_id) const;
    
    // Get file descriptor for an index
    int getFileFd(u32 index_id) const;
    
    // Persist catalog to disk
    void persist();
    
    // Load catalog from disk
    void load();
    
    // Register a new index
    void registerIndex(u32 index_id, u64 max_pages, int file_fd = -1);
    
    // Clear all entries (for cleanup on reinitialization)
    void clear();
};

struct BufferManager {
    static const u64 mb = 1024ull * 1024;
    static const u64 gb = 1024ull * 1024 * 1024;
    u64 virtSize;
    u64 physSize;
    u64 virtCount;
    u64 physCount;
    struct exmap_user_interface* exmapInterface[maxWorkerThreads];

    enum class HashMode { Array1Access = 0, Unordered = 1, OpenAddress = 2, Array = 3, Lockfree = 4, Array2Level = 5, Array3Level = 6 };

    bool useTraditional;
    bool useMmapOSPageacche = false;
    bool useExmap;
    HashMode hashMode;
    unsigned numThreads;
    int blockfd;
    int exmapfd;

    static constexpr u64 invalidFrame = std::numeric_limits<u64>::max();
    static constexpr PID invalidPID = std::numeric_limits<PID>::max();
    static constexpr u32 freePartitionCount = 64;

    char cachline_pad0[64];
    ResidentPageSet residentSet;

    std::atomic<u64> physUsedCount;
    char cachline_pad1[64];
    std::atomic<u64> allocCount;
    char cachline_pad2[64];
    std::atomic<u64> readCount;
    char cachline_pad3[64];
    std::atomic<u64> writeCount;
    char cachline_pad4[64];
    std::atomic<u64> holePunchCount;  // Counter for translation table hole-punching operations
    char cachline_pad5[64];

    // Index Catalog for per-index allocation tracking
    std::unique_ptr<IndexCatalog> indexCatalog;
    std::unordered_map<u32, std::unique_ptr<PIDAllocator>> perIndexAllocators;
    mutable std::shared_mutex catalogMutex;

    Page* virtMem;
    Page* frameMem;
    PageState* pageState;                                    // Single-level array (for non-Array2Level/Array3Level modes)
    TwoLevelPageStateArray* pageState2Level; // Two-level array (for Array2Level mode)
    std::unique_ptr<ThreeLevelPageStateArray> pageState3Level; // Three-level array (for Array3Level mode)
    std::unique_ptr<std::atomic<u64>[]> pidToFrameArray;
    std::atomic<u32>* translationRefCounts;  // Reference counts for OS page groups (lazily allocated via mmap)
    u64 numRefCountGroups;  // Number of OS page groups in translation array
    u64 translationOSPageSize;  // OS page size used for translation array (4KB or 2MB based on huge pages)
    static constexpr u32 REF_COUNT_LOCK_BIT = 0x80000000u;  // Highest bit used as lock
    static constexpr u32 REF_COUNT_MASK = 0x7FFFFFFFu;      // Lower 31 bits for count
    static constexpr u64 NUM_LOCK_SHARDS = 64;
    std::unique_ptr<pthread_rwlock_t[]> pid_locks;
    std::unique_ptr<std::atomic<PID>[]> frameToPidArray;
    std::unique_ptr<PartitionedMap<PID, u64>> pidToFrameHash;
    std::unique_ptr<PartitionedMap<u64, PID>> frameToPidHash;
    std::array<FreePartition, freePartitionCount> freePartitions;
    u32 partitionFrameSpan;
    u64 batch;

    PageState& getPageState(PID pid) {
        //if (hashMode == HashMode::Array2Level) {
            return pageState2Level->get(pid);
        //}
        // if (hashMode == HashMode::Array3Level) {
        //     return pageState3Level->get(pid);
        // }
        //return pageState[pid];
    }
    u64 pid_lock_shard(PID pid) {
        return pid % NUM_LOCK_SHARDS;
    }

    void flushAll();

    BufferManager(unsigned nthreads = 1);
    ~BufferManager();

    Page* fixX(PID pid);
    void unfixX(PID pid);
    Page* fixS(PID pid);
    void unfixS(PID pid);

    bool isValidPtr(void* page);
    PID toPID(void* page);
    Page* toPtr(PID pid);
    Page* residentPtr(PID pid);
    Page* residentPtr(PID pid, u64 stateAndVersion);
    Page* residentPtrCalico(PID pid, u64 stateAndVersion);
    Page* residentPtrHash(PID pid, u64 stateAndVersion);
    Page* preparePageForWrite(PID pid);

    void ensureFreePages();
    Page* allocPage(PIDAllocator* allocator = nullptr);
    Page* allocPageForIndex(u32 indexId, PIDAllocator* allocator = nullptr);
    void updateAllocCountSnapshot(u64 value);
    
    // Get or create a PIDAllocator for an index
    PIDAllocator* getOrCreateAllocatorForIndex(u32 index_id, u64 max_pages = 0);
    Page* handleFault(PID pid);
    void readPage(PID pid, Page* dest);
    void evict();
    void forceEvictPortion(float portion = 0.5);  // Force eviction of a portion of buffer pool for testing
    void prefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);
    void prefetchPagesSingleLevel(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);
    void prefetchPages2Level(const PID* pages, int n_pages, const u32* offsets_within_pages = nullptr);

    Page* acquireFrameForPid(PID pid);
    void releaseFrame(PID pid);
    u32 popFreeFrame();

    u32 partitionForFrame(u64 frame) const;
    void lockPartition(u32 partition);
    void unlockPartition(u32 partition);
    
    IOInterface& getIOInterface();

    u64 countZeroRefCountGroups();
    
    // Multi-index support methods (for Array2Level mode)
    /**
     * Get the file descriptor and local page ID for a given global PID.
     * In Array2Level mode, extracts index_id from PID and looks up the corresponding file descriptor.
     * @param pid Global page ID
     * @param localPageId Output parameter for local page ID within the index
     * @return File descriptor for the index, or blockfd if not using Array2Level
     */
    int getFileDescriptorForPID(PID pid, PID& localPageId);
    
    /**
     * Register a new index with the buffer manager.
     * @param indexId The index ID (0-65535)
     * @param maxPages Maximum number of pages for this index
     * @param initialAllocCount Initial allocation count from catalog recovery (default: 0)
     * @param fileFd File descriptor for this index's data file (optional, for caching)
     */
    void registerIndex(u32 indexId, u64 maxPages, u64 initialAllocCount = 0, int fileFd = -1) {
        // Register with simple IndexCatalog (for fd tracking)
        if (indexCatalog) {
            indexCatalog->registerIndex(indexId, maxPages, fileFd);
        }
        
        if (hashMode != HashMode::Array2Level || !pageState2Level) {
            // In non-Array2Level modes, we don't need explicit index registration
            // The single translation array handles all PIDs
            return;
        }
        pageState2Level->registerIndex(indexId, maxPages, initialAllocCount, fileFd);
    }
    
    /**
     * Unregister an index and free its translation array.
     */
    void unregisterIndex(u32 indexId) {
        if (hashMode != HashMode::Array2Level || !pageState2Level) {
            return;
        }
        pageState2Level->unregisterIndex(indexId);
    }
    
    /**
     * Unregister all non-zero indexes (for reset/close).
     * This allows caliby.open() to be called again with a fresh state.
     */
    void unregisterAllNonZero() {
        if (hashMode != HashMode::Array2Level || !pageState2Level) {
            return;
        }
        
        // First, clear all per-index allocators (except index 0)
        {
            std::unique_lock<std::shared_mutex> lock(catalogMutex);
            std::vector<u32> toRemove;
            for (auto& [indexId, allocator] : perIndexAllocators) {
                if (indexId != 0) {
                    toRemove.push_back(indexId);
                }
            }
            for (u32 indexId : toRemove) {
                perIndexAllocators.erase(indexId);
            }
        }
        
        // Clear resident set entries for non-zero indexes
        if (useTraditional) {
            residentSet.removeAllForIndexes(1);  // Remove all pages for index >= 1
        }
        
        // Then unregister from TwoLevelPageStateArray
        pageState2Level->unregisterAllNonZero();
    }
    
    /**
     * Check if an index is registered.
     */
    bool isIndexRegistered(u32 indexId) const {
        if (hashMode != HashMode::Array2Level || !pageState2Level) {
            // In non-Array2Level modes, all indexes are implicitly available
            return true;
        }
        return pageState2Level->isIndexRegistered(indexId);
    }
    
    /**
     * Get per-index translation array (for advanced usage).
     */
    IndexTranslationArray* getIndexArray(u32 indexId) const {
        if (hashMode != HashMode::Array2Level || !pageState2Level) {
            return nullptr;
        }
        return pageState2Level->getIndexArray(indexId);
    }
    
    /**
     * Check if multi-index PID encoding is enabled (only for Array2Level mode).
     */
    bool supportsMultiIndexPIDs() const {
        return hashMode == HashMode::Array2Level;
    }
    
    /**
     * Compose a PID from index_id and local_page_id.
     */
    static PID makeIndexPID(u32 indexId, u32 localPageId) {
        return TwoLevelPageStateArray::makePID(indexId, localPageId);
    }
    
    /**
     * Extract index_id from a PID.
     */
    static u32 getIndexIdFromPID(PID pid) {
        return TwoLevelPageStateArray::getIndexId(pid);
    }
    
    /**
     * Extract local_page_id from a PID.
     */
    static u32 getLocalPageIdFromPID(PID pid) {
        return TwoLevelPageStateArray::getLocalPageId(pid);
    }
    
    /**
     * Persist all index translation array capacities to the catalog.
     * Called before shutdown to ensure proper recovery on restart.
     */
    void persistIndexCapacities();
};
typedef u64 KeyType;

extern BufferManager* bm_ptr;
// Flag to track if the system has been closed (index arrays unregistered)
extern bool system_closed;
// Define a convenience reference to avoid changing all code that uses bm
#define bm (*bm_ptr)

struct OLCRestartException {};

template <class T>
struct GuardO {
    PID pid;
    T* ptr;
    u64 version;
    static const u64 moved = ~0ull;
    IndexTranslationArray* arr = nullptr;

    // constructor
    explicit GuardO(u64 pid) : pid(pid), ptr(nullptr), arr(nullptr) { init(); }

    // Specialized constructor with IndexTranslationArray to bypass TLS cache
    explicit GuardO(u64 pid, IndexTranslationArray* arr) : pid(pid), ptr(nullptr), arr(arr) { initWithArray(arr); }

    template <class T2>
    GuardO(u64 pid, GuardO<T2>& parent) {
        parent.checkVersionAndRestart();
        this->pid = pid;
        ptr = nullptr;
        init();
    }

    GuardO(GuardO&& other) {
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
        arr = other.arr;
        other.pid = moved;
        other.ptr = nullptr;
        other.arr = nullptr;
    }

    void init() {
        assert(pid != moved);
        PageState& ps = bm.getPageState(pid);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            u64 v = ps.stateAndVersion.load();
            switch (PageState::getState(v)) {
                case PageState::Marked: {
                    u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                    if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                        ptr = reinterpret_cast<T*>(bm.residentPtr(pid, newV));
                        version = newV;
                        return;
                    }
                    break;
                }
                case PageState::Locked:
                    break;
                case PageState::Evicted:
                    if (ps.tryLockX(v)) {
                        ptr = reinterpret_cast<T*>(bm.handleFault(pid));
                        bm.unfixX(pid);
                        bm.ensureFreePages();
                    }
                    break;
                default:
#if defined(CALICO_SPECIALIZATION_CALICO)
                    ptr = reinterpret_cast<T*>(bm.residentPtrCalico(pid, v));
#elif defined(CALICO_SPECIALIZATION_HASH)
                    ptr = reinterpret_cast<T*>(bm.residentPtrHash(pid, v));
#else
                    ptr = reinterpret_cast<T*>(bm.residentPtr(pid, v));
#endif
                    version = v;
                    return;
            }
            yield(repeatCounter);
        }
    }

    void initWithArray(IndexTranslationArray* arr) {
        assert(pid != moved);
        // Fallback to regular init if arr is null (for non-Array2Level modes)
        // if (arr == nullptr) {
        //     init();
        //     return;
        // }
        // Extract local page ID from the global PID
        u32 localPageId = static_cast<u32>(pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK);
        PageState& ps = arr->get(localPageId);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            u64 v = ps.stateAndVersion.load();
            switch (PageState::getState(v)) {
                case PageState::Marked: {
                    u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                    if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                        ptr = reinterpret_cast<T*>(bm.residentPtr(pid, newV));
                        version = newV;
                        return;
                    }
                    break;
                }
                case PageState::Locked:
                    break;
                case PageState::Evicted:
                    if (ps.tryLockX(v)) {
                        ptr = reinterpret_cast<T*>(bm.handleFault(pid));
                        bm.unfixX(pid);
                        bm.ensureFreePages();
                    }
                    break;
                default:
#if defined(CALICO_SPECIALIZATION_CALICO)
                    ptr = reinterpret_cast<T*>(bm.residentPtrCalico(pid, v));
#elif defined(CALICO_SPECIALIZATION_HASH)
                    ptr = reinterpret_cast<T*>(bm.residentPtrHash(pid, v));
#else
                    ptr = reinterpret_cast<T*>(bm.residentPtr(pid, v));
#endif
                    version = v;
                    return;
            }
            yield(repeatCounter);
        }
    }

    // move assignment operator
    GuardO& operator=(GuardO&& other) {
        if (pid != moved) checkVersionAndRestart();
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
        arr = other.arr;
        other.pid = moved;
        other.ptr = nullptr;
        other.arr = nullptr;
        return *this;
    }

    // assignment operator
    GuardO& operator=(const GuardO&) = delete;

    // copy constructor
    GuardO(const GuardO&) = delete;

    void checkVersionAndRestart() {
        if (pid != moved) {
            // PageState& ps = bm.getPageState(pid);
            u32 localPageId = static_cast<u32>(pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK);
            PageState& ps = arr ? arr->get(localPageId) : bm.getPageState(pid);

            u64 stateAndVersion = ps.stateAndVersion.load();
            if (version == stateAndVersion)  // fast path, nothing changed
                return;
            if ((stateAndVersion << 8) == (version << 8)) {  // same version
                u64 state = PageState::getState(stateAndVersion);
                if (state > PageState::Unlocked && state <= PageState::MaxShared) return;  // ignore shared locks
                if (state == PageState::Marked)
                    if (ps.stateAndVersion.compare_exchange_weak(
                            stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                        return;  // mark cleared
            }
            if (std::uncaught_exceptions() == 0) throw OLCRestartException();
        }
    }

    // destructor
    ~GuardO() noexcept(false) { checkVersionAndRestart(); }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        checkVersionAndRestart();
        pid = moved;
        ptr = nullptr;
    }
};

// Guard that allows relaxed checking of version changes.
// Useful the data structures can tolerate inconsistent reads, e.g. graph nodes in ANNS index.
template <class T>
struct GuardORelaxed {
    PID pid;
    T* ptr;
    u64 version;
    static const u64 moved = ~0ull;
    IndexTranslationArray* arr;
    // constructor
    explicit GuardORelaxed(u64 pid) : pid(pid), ptr(nullptr), arr(nullptr) { init(); }

    // Specialized constructor with IndexTranslationArray to bypass TLS cache
    explicit GuardORelaxed(u64 pid, IndexTranslationArray* arr) : pid(pid), ptr(nullptr), arr(arr) { initWithArray(arr); }

    template <class T2>
    GuardORelaxed(u64 pid, GuardO<T2>& parent) {
        // we do not need to check the parent's version here, as we are relaxed
        //parent.checkVersionAndRestart();
        this->pid = pid;
        ptr = nullptr;
        init();
    }

    template <class T2>
    GuardORelaxed(GuardO<T2>&& other) {
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
    }


    template <class T2>
    GuardORelaxed(u64 pid, GuardORelaxed<T2>& parent) {
        this->pid = pid;
        ptr = nullptr;
        init();
    }

    GuardORelaxed(GuardORelaxed&& other) {
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
        arr = other.arr;
        other.pid = moved;
        other.ptr = nullptr;
        other.arr = nullptr;
    }


    void init() {
        assert(pid != moved);
        PageState& ps = bm.getPageState(pid);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            u64 v = ps.stateAndVersion.load();
            switch (PageState::getState(v)) {
                // case PageState::Marked: {
                //     u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                //     if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                //         ptr = reinterpret_cast<T*>(bm.residentPtr(pid, ps, newV));
                //         version = newV;
                //         return;
                //     }
                //     break;
                // }
                case PageState::Evicted:
                    if (ps.tryLockX(v)) {
                        try {
                            ptr = reinterpret_cast<T*>(bm.handleFault(pid));
                            bm.unfixX(pid);
                            bm.ensureFreePages();
                        } catch (...) {
                            // Unlock page on failure to prevent livelock
                            ps.unlockXEvicted();
                            throw;
                        }
                    }
                    break;
                case PageState::Marked:
                case PageState::Locked:// we can live with stale data here, fallthrough
                default:
#if defined(CALICO_SPECIALIZATION_CALICO)
                    ptr = reinterpret_cast<T*>(bm.residentPtrCalico(pid, v));
#elif defined(CALICO_SPECIALIZATION_HASH)
                    ptr = reinterpret_cast<T*>(bm.residentPtrHash(pid, v));
#else
                    ptr = reinterpret_cast<T*>(bm.residentPtr(pid, v));
#endif
                    version = v;
                    return;
            }
            yield(repeatCounter);
        }
    }

    void initWithArray(IndexTranslationArray* arr) {
        assert(pid != moved);
        // Extract local page ID from the global PID
        u32 localPageId = static_cast<u32>(pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK);
        PageState& ps = arr->get(localPageId);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            u64 v = ps.stateAndVersion.load();
            switch (PageState::getState(v)) {
                // case PageState::Marked: {
                //     u64 newV = PageState::sameVersion(v, PageState::Unlocked);
                //     if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                //         ptr = reinterpret_cast<T*>(bm.residentPtr(pid, newV));
                //         version = newV;
                //         return;
                //     }
                //     break;
                // }
                case PageState::Evicted:
                    if (ps.tryLockX(v)) {
                        ptr = reinterpret_cast<T*>(bm.handleFault(pid));
                        bm.unfixX(pid);
                        bm.ensureFreePages();
                    }
                    break;
                case PageState::Marked:
                case PageState::Locked:
                default:
#if defined(CALICO_SPECIALIZATION_CALICO)
                    ptr = reinterpret_cast<T*>(bm.residentPtrCalico(pid, v));
#elif defined(CALICO_SPECIALIZATION_HASH)
                    ptr = reinterpret_cast<T*>(bm.residentPtrHash(pid, v));
#else
                    ptr = reinterpret_cast<T*>(bm.residentPtr(pid, v));
#endif
                    version = v;
                    return;
            }
            yield(repeatCounter);
        }
    }

    // move assignment operator
    GuardORelaxed& operator=(GuardORelaxed&& other) {
        if (pid != moved) checkVersionAndRestart();
        pid = other.pid;
        ptr = other.ptr;
        version = other.version;
        arr = other.arr;
        other.pid = moved;
        other.ptr = nullptr;
        other.arr = nullptr;
        return *this;
    }

    // assignment operator
    GuardORelaxed& operator=(const GuardORelaxed&) = delete;

    // copy constructor
    GuardORelaxed(const GuardORelaxed&) = delete;

    void checkVersionAndRestart() {
        if (pid != moved) {
            u32 localPageId = static_cast<u32>(pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK);
            PageState& ps = arr ? arr->get(localPageId) : bm.getPageState(pid);

            u64 stateAndVersion = ps.stateAndVersion.load();
            if (version == stateAndVersion)  // fast path, nothing changed
                return;
            u64 oldOCCVersion = PageState::extractVersion(version);
            u64 currentOCCVersion = PageState::extractVersion(stateAndVersion);
            u64 oldFrameValue = PageState::getFrameValue(version);
            u64 currentFrameValue = PageState::getFrameValue(stateAndVersion);
            if (oldFrameValue != currentFrameValue) {
                // frame changed, must restart. 
                // This is because we might be reading data that is from a different pages which suggests a significant change.
                if (std::uncaught_exceptions() == 0) throw OLCRestartException();
            }
            if (oldOCCVersion == currentOCCVersion) {  // same version
                u64 state = PageState::getState(stateAndVersion);
                if (state > PageState::Unlocked && state <= PageState::MaxShared) return;  // ignore shared locks
                if (state == PageState::Marked)
                    if (ps.stateAndVersion.compare_exchange_weak(
                            stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                        return;  // mark cleared
            }
            // it is fine if the version changed, we are relaxed
        }
    }

    // destructor
    ~GuardORelaxed() noexcept(false) { 
        //checkVersionAndRestart(); 
    }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        checkVersionAndRestart();
        pid = moved;
        ptr = nullptr;
    }
};


template <class T>
struct GuardX {
    PID pid;
    T* ptr;
    static const u64 moved = ~0ull;

    // constructor
    GuardX() : pid(moved), ptr(nullptr) {}

    // constructor
    explicit GuardX(u64 pid) : pid(pid) {
        ptr = reinterpret_cast<T*>(bm.fixX(pid));
        ptr->dirty = true;
    }

    explicit GuardX(GuardO<T>&& other) {
        assert(other.pid != moved);
        for (u64 repeatCounter = 0;; repeatCounter++) {
            PageState& ps = bm.getPageState(other.pid);
            u64 stateAndVersion = ps.stateAndVersion;
            if ((stateAndVersion << 8) != (other.version << 8)) throw OLCRestartException();
            u64 state = PageState::getState(stateAndVersion);
            if ((state == PageState::Unlocked) || (state == PageState::Marked)) {
                if (ps.tryLockX(stateAndVersion)) {
                    pid = other.pid;
                    ptr = other.ptr;
                    ptr->dirty = true;
                    other.pid = moved;
                    other.ptr = nullptr;
                    return;
                }
            }
            yield(repeatCounter);
        }
    }

    // assignment operator
    GuardX& operator=(const GuardX&) = delete;

    // move assignment operator
    GuardX& operator=(GuardX&& other) {
        if (pid != moved) {
            bm.unfixX(pid);
        }
        pid = other.pid;
        ptr = other.ptr;
        other.pid = moved;
        other.ptr = nullptr;
        return *this;
    }

    // copy constructor
    GuardX(const GuardX&) = delete;

    // destructor
    ~GuardX() {
        if (pid != moved) bm.unfixX(pid);
    }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        if (pid != moved) {
            bm.unfixX(pid);
            pid = moved;
        }
    }
};

template <class T>
struct AllocGuard : public GuardX<T> {
    // Main constructor with PIDAllocator and optional constructor parameters
    template <typename... Params>
    AllocGuard(PIDAllocator* allocator, Params&&... params) {
        u32 indexId = allocator ? allocator->index_id : 0;
        GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPageForIndex(indexId, allocator));
        new (GuardX<T>::ptr) T(std::forward<Params>(params)...);
        GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
        GuardX<T>::ptr->dirty = true;  // Mark newly allocated page as dirty
    }

    // Default constructor (no allocator, no parameters) - uses index 0
    AllocGuard() {
        GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPage(nullptr));
        new (GuardX<T>::ptr) T();
        GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
        GuardX<T>::ptr->dirty = true;  // Mark newly allocated page as dirty
    }
};

template <class T>
struct GuardS {
    PID pid;
    T* ptr;
    static const u64 moved = ~0ull;

    // constructor
    explicit GuardS(u64 pid) : pid(pid) { ptr = reinterpret_cast<T*>(bm.fixS(pid)); }

    GuardS(GuardO<T>&& other) {
        assert(other.pid != moved);
        if (bm.getPageState(other.pid).tryLockS(other.version)) {  // XXX: optimize?
            pid = other.pid;
            ptr = other.ptr;
            other.pid = moved;
            other.ptr = nullptr;
        } else {
            throw OLCRestartException();
        }
    }

    GuardS(GuardS&& other) {
        if (pid != moved) bm.unfixS(pid);
        pid = other.pid;
        ptr = other.ptr;
        other.pid = moved;
        other.ptr = nullptr;
    }

    // assignment operator
    GuardS& operator=(const GuardS&) = delete;

    // move assignment operator
    GuardS& operator=(GuardS&& other) {
        if (pid != moved) bm.unfixS(pid);
        pid = other.pid;
        ptr = other.ptr;
        other.pid = moved;
        other.ptr = nullptr;
        return *this;
    }

    // copy constructor
    GuardS(const GuardS&) = delete;

    // destructor
    ~GuardS() {
        if (pid != moved) bm.unfixS(pid);
    }

    T* operator->() {
        assert(pid != moved);
        return ptr;
    }

    void release() {
        if (pid != moved) {
            bm.unfixS(pid);
            pid = moved;
        }
    }
};

u64 envOr(const char* env, u64 value);
double envOrDouble(const char* env, double value);

//---------------------------------------------------------------------------

struct BTreeNode;

struct BTreeNodeHeader {
    static const unsigned underFullSize = (pageSize / 2) + (pageSize / 4);  // merge nodes more empty
    static const u64 noNeighbour = ~0ull;

    struct FenceKeySlot {
        u16 offset;
        u16 len;
    };

    bool dirty;
    union {
        PID upperInnerNode;              // inner
        PID nextLeafNode = noNeighbour;  // leaf
    };

    bool hasRightNeighbour() { return nextLeafNode != noNeighbour; }

    FenceKeySlot lowerFence = {0, 0};  // exclusive
    FenceKeySlot upperFence = {0, 0};  // inclusive

    bool hasLowerFence() { return !!lowerFence.len; };

    u16 count = 0;
    bool isLeaf;
    u16 spaceUsed = 0;
    u16 dataOffset = static_cast<u16>(pageSize);
    u16 prefixLen = 0;

    static const unsigned hintCount = 16;
    u32 hint[hintCount];
    u32 padding;

    BTreeNodeHeader(bool isLeaf) : isLeaf(isLeaf) {}
    ~BTreeNodeHeader() {}
};

struct BTreeNode : public BTreeNodeHeader {
    struct Slot {
        u16 offset;
        u16 keyLen;
        u16 payloadLen;
        union {
            u32 head;
            u8 headBytes[4];
        };
    } __attribute__((packed));
    union {
        Slot slot[(pageSize - sizeof(BTreeNodeHeader)) / sizeof(Slot)];  // grows from front
        u8 heap[pageSize - sizeof(BTreeNodeHeader)];                     // grows from back
    };

    static constexpr unsigned maxKVSize = ((pageSize - sizeof(BTreeNodeHeader) - (2 * sizeof(Slot)))) / 4;

    BTreeNode(bool isLeaf) : BTreeNodeHeader(isLeaf) { dirty = true; }

    u8* ptr() { return reinterpret_cast<u8*>(this); }
    bool isInner() { return !isLeaf; }
    std::span<u8> getLowerFence() { return {ptr() + lowerFence.offset, lowerFence.len}; }
    std::span<u8> getUpperFence() { return {ptr() + upperFence.offset, upperFence.len}; }
    u8* getPrefix() { return ptr() + lowerFence.offset; }  // any key on page is ok

    unsigned freeSpace() { return dataOffset - (reinterpret_cast<u8*>(slot + count) - ptr()); }
    unsigned freeSpaceAfterCompaction() { return pageSize - (reinterpret_cast<u8*>(slot + count) - ptr()) - spaceUsed; }

    bool hasSpaceFor(unsigned keyLen, unsigned payloadLen);

    u8* getKey(unsigned slotId) { return ptr() + slot[slotId].offset; }
    std::span<u8> getPayload(unsigned slotId) {
        return {ptr() + slot[slotId].offset + slot[slotId].keyLen, slot[slotId].payloadLen};
    }
    PID getChild(unsigned slotId);

    unsigned spaceNeeded(unsigned keyLen, unsigned payloadLen) {
        return sizeof(Slot) + (keyLen - prefixLen) + payloadLen;
    }

    void makeHint();
    void updateHint(unsigned slotId);
    void searchHint(u32 keyHead, u16& lowerOut, u16& upperOut);
    u16 lowerBound(std::span<u8> skey, bool& foundExactOut);
    u16 lowerBound(std::span<u8> key);
    void insertInPage(std::span<u8> key, std::span<u8> payload);
    bool removeSlot(unsigned slotId);
    bool removeInPage(std::span<u8> key);
    void copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src);
    void compactify();
    bool mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right);
    void storeKeyValue(u16 slotId, std::span<u8> skey, std::span<u8> payload);
    void copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount);
    void copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot);
    void insertFence(FenceKeySlot& fk, std::span<u8> key);
    void setFences(std::span<u8> lower, std::span<u8> upper);
    void splitNode(BTreeNode* parent, unsigned sepSlot, std::span<u8> sep, PIDAllocator* allocator);

    struct SeparatorInfo {
        unsigned len;      // len of new separator
        unsigned slot;     // slot at which we split
        bool isTruncated;  // if true, we truncate the separator taking len bytes from slot+1
    };

    unsigned commonPrefix(unsigned slotA, unsigned slotB);
    SeparatorInfo findSeparator(bool splitOrdered);
    void getSep(u8* sepKeyOut, SeparatorInfo info);
    PID lookupInner(std::span<u8> key);
};

static_assert(sizeof(BTreeNode) == pageSize, "btree node size problem");

static const u64 metadataPageId = 0;

struct HNSWMetaInfo {
    static constexpr u64 magic = 0x484E53574D455441ULL;  // "HNSWMETA"
    u64 magic_value = 0;
    PID metadata_pid = BufferManager::invalidPID;
    PID base_pid = BufferManager::invalidPID;
    u64 max_elements = 0;
    u64 dim = 0;
    u64 M = 0;
    u64 ef_construction = 0;
    u64 max_level = 0;
    std::atomic<u64> alloc_count{0};  // Per-index allocation counter
    u8 valid = 0;
    u8 reserved[6] = {0};

    bool isValid() const {
        return valid != 0 && magic_value == magic && metadata_pid != BufferManager::invalidPID &&
               base_pid != BufferManager::invalidPID;
    }

    void clear() {
        magic_value = 0;
        metadata_pid = BufferManager::invalidPID;
        base_pid = BufferManager::invalidPID;
        max_elements = 0;
        dim = 0;
        M = 0;
        ef_construction = 0;
        max_level = 0;
        valid = 0;
        memset(reserved, 0, sizeof(reserved));
    }
};

// IVF+PQ meta info for recovery
struct IVFPQMetaInfoCompact {
    static constexpr u64 magic = 0x49564650514D4554ULL;  // "IVFPQMET"
    
    u64 magic_value = 0;
    u8 valid = 0;
    u8 is_trained = 0;
    u8 reserved1[6] = {0};
    
    // Core parameters
    u32 dim = 0;
    u32 num_clusters = 0;        // K
    u32 num_subquantizers = 0;   // M
    u32 subvector_dim = 0;       // dim / M
    
    // Page locations
    PID metadata_pid = BufferManager::invalidPID;
    PID centroids_base_pid = BufferManager::invalidPID;
    PID invlist_dir_base_pid = BufferManager::invalidPID;
    PID codebook_base_pid = BufferManager::invalidPID;
    
    // State
    u64 max_elements = 0;
    std::atomic<u64> num_vectors{0};
    std::atomic<u64> last_train_count{0};
    u32 retrain_interval = 0;
    u8 reserved2[4] = {0};
    
    bool isValid() const {
        return valid != 0 && magic_value == magic && metadata_pid != BufferManager::invalidPID;
    }
    
    void clear() {
        magic_value = 0;
        valid = 0;
        is_trained = 0;
        memset(reserved1, 0, sizeof(reserved1));
        dim = 0;
        num_clusters = 0;
        num_subquantizers = 0;
        subvector_dim = 0;
        metadata_pid = BufferManager::invalidPID;
        centroids_base_pid = BufferManager::invalidPID;
        invlist_dir_base_pid = BufferManager::invalidPID;
        codebook_base_pid = BufferManager::invalidPID;
        max_elements = 0;
        num_vectors.store(0);
        last_train_count.store(0);
        retrain_interval = 0;
        memset(reserved2, 0, sizeof(reserved2));
    }
};

struct MetaDataPage {
    bool dirty;
    u8 padding[7] = {0};
    HNSWMetaInfo hnsw_meta;
    IVFPQMetaInfoCompact ivfpq_meta;
    u64 alloc_count_snapshot = 0;
    PID roots[(pageSize - sizeof(dirty) - sizeof(padding) - sizeof(HNSWMetaInfo) - sizeof(IVFPQMetaInfoCompact) - sizeof(alloc_count_snapshot)) /
              sizeof(PID)];

    PID getRoot(unsigned slot) { return roots[slot]; }

    u64 getAllocCountSnapshot() const { return alloc_count_snapshot; }

    void setAllocCountSnapshot(u64 value) { alloc_count_snapshot = value; }
};

void flush_system();

static_assert(sizeof(MetaDataPage) <= pageSize, "MetaDataPage must fit within a single page");

struct BTree {
   private:
    void trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, std::span<u8> key, unsigned payloadLen);
    void ensureSpace(BTreeNode* toSplit, std::span<u8> key, unsigned payloadLen);

   public:
    unsigned slotId;
    std::atomic<bool> splitOrdered;
    PIDAllocator pidAllocator;

    BTree();                              // Create new BTree with new slotId
    explicit BTree(unsigned existingSlotId); // Recover existing BTree from slotId
    ~BTree();
    
    // Get the slot ID for persistence
    unsigned getSlotId() const { return slotId; }

    GuardO<BTreeNode> findLeafO(std::span<u8> key);
    GuardS<BTreeNode> findLeafS(std::span<u8> key);

    int lookup(std::span<u8> key, u8* payloadOut, unsigned payloadOutSize) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return -1;

                // key found, copy payload
                 memcpy(payloadOut, node->getPayload(pos).data(), std::min(static_cast<unsigned>(node->slot[pos].payloadLen), payloadOutSize));
                return node->slot[pos].payloadLen;
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    template <class Fn>
    bool lookup(std::span<u8> key, Fn fn) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return false;

                // key found
                fn(node->getPayload(pos));
                return true;
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    void insert(std::span<u8> key, std::span<u8> payload);
    bool remove(std::span<u8> key);

    template <class Fn>
    bool updateInPlace(std::span<u8> key, Fn fn) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return false;

                {
                    GuardX<BTreeNode> nodeLocked(std::move(node));
                    fn(nodeLocked->getPayload(pos));
                    return true;
                }
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    /**
     * Update a key's payload out-of-place (delete + insert).
     * Returns: 0 = success, -1 = key not found, -2 = no space in leaf for new payload
     * This is useful when the new payload size differs from the old one.
     */
    int updateOutOfPlace(std::span<u8> key, std::span<u8> newPayload) {
        // First check if key exists and if we have space
        for (u64 repeatCounter = 0;; repeatCounter++) {
            try {
                GuardO<BTreeNode> node = findLeafO(key);
                bool found;
                unsigned pos = node->lowerBound(key, found);
                if (!found) return -1;  // Key not found
                
                // Calculate space needed for new entry vs freed space from old entry
                unsigned oldKeyLen = node->slot[pos].keyLen + node->prefixLen;
                unsigned oldPayloadLen = node->slot[pos].payloadLen;
                unsigned oldSpaceUsed = sizeof(BTreeNode::Slot) + node->slot[pos].keyLen + oldPayloadLen;
                unsigned newSpaceNeeded = node->spaceNeeded(key.size(), newPayload.size());
                
                // Check if after deletion we have space for the new entry
                unsigned freeAfterDelete = node->freeSpaceAfterCompaction() + oldSpaceUsed;
                if (freeAfterDelete < newSpaceNeeded) {
                    return -2;  // No space even after deletion
                }
                
                // Acquire write lock and perform delete + insert atomically
                {
                    GuardX<BTreeNode> nodeLocked(std::move(node));
                    nodeLocked->removeSlot(pos);
                    nodeLocked->insertInPage(key, newPayload);
                    return 0;  // Success
                }
            } catch (const OLCRestartException&) {
                yield(repeatCounter);
            }
        }
    }

    template <class Fn>
    void scanAsc(std::span<u8> key, Fn fn) {
        GuardS<BTreeNode> node = findLeafS(key);
        bool found;
        unsigned pos = node->lowerBound(key, found);
        for (u64 repeatCounter = 0;; repeatCounter++) {  // XXX
            if (pos < node->count) {
                if (!fn(*node.ptr, pos)) return;
                pos++;
            } else {
                if (!node->hasRightNeighbour()) return;
                pos = 0;
                node = GuardS<BTreeNode>(node->nextLeafNode);
            }
        }
    }

    template <class Fn>
    void scanDesc(std::span<u8> key, Fn fn) {
        GuardS<BTreeNode> node = findLeafS(key);
        bool exactMatch;
        int pos = node->lowerBound(key, exactMatch);
        if (pos == node->count) {
            pos--;
            exactMatch = true;  // XXX:
        }
        for (u64 repeatCounter = 0;; repeatCounter++) {  // XXX
            while (pos >= 0) {
                if (!fn(*node.ptr, pos, exactMatch)) return;
                pos--;
            }
            if (!node->hasLowerFence()) return;
            node = findLeafS(node->getLowerFence());
            pos = node->count - 1;
        }
    }
};

template <class Record>
struct vmcacheAdapter {
    BTree tree;

   public:
    void scan(const typename Record::Key& key,
              const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb,
              std::function<void()> reset_if_scan_failed_cb) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        u8 kk[Record::maxFoldLength()];
        tree.scanAsc({k, l}, [&](BTreeNode& node, unsigned slot) {
            memcpy(kk, node.getPrefix(), node.prefixLen);
            memcpy(kk + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
            typename Record::Key typedKey;
            Record::unfoldKey(kk, typedKey);
            return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
        });
    }
    // -------------------------------------------------------------------------------------
    void scanDesc(const typename Record::Key& key,
                  const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb,
                  std::function<void()> reset_if_scan_failed_cb) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        u8 kk[Record::maxFoldLength()];
        bool first = true;
        tree.scanDesc({k, l}, [&](BTreeNode& node, unsigned slot, bool exactMatch) {
            if (first) {  // XXX: hack
                first = false;
                if (!exactMatch) return true;
            }
            memcpy(kk, node.getPrefix(), node.prefixLen);
            memcpy(kk + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
            typename Record::Key typedKey;
            Record::unfoldKey(kk, typedKey);
            return found_record_cb(typedKey, *reinterpret_cast<const Record*>(node.getPayload(slot).data()));
        });
    }
    // -------------------------------------------------------------------------------------
    void insert(const typename Record::Key& key, const Record& record) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        tree.insert({k, l}, {(u8*)(&record), sizeof(Record)});
    }
    // -------------------------------------------------------------------------------------
    template <class Fn>
    void lookup1(const typename Record::Key& key, Fn fn) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        bool succ =
            tree.lookup({k, l}, [&](std::span<u8> payload) { fn(*reinterpret_cast<const Record*>(payload.data())); });
        assert(succ);
    }
    // -------------------------------------------------------------------------------------
    template <class Fn>
    void update1(const typename Record::Key& key, Fn fn) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        tree.updateInPlace({k, l}, [&](std::span<u8> payload) { fn(*reinterpret_cast<Record*>(payload.data())); });
    }
    // -------------------------------------------------------------------------------------
    // Returns false if the record was not found
    bool erase(const typename Record::Key& key) {
        u8 k[Record::maxFoldLength()];
        u16 l = Record::foldKey(k, key);
        return tree.remove({k, l});
    }
    // -------------------------------------------------------------------------------------
    template <class Field>
    Field lookupField(const typename Record::Key& key, Field Record::* f) {
        Field value;
        lookup1(key, [&](const Record& r) { value = r.*f; });
        return value;
    }

    u64 count() {
        u64 cnt = 0;
        tree.scanAsc({(u8*)nullptr, 0}, [&](BTreeNode& node, unsigned slot) {
            cnt++;
            return true;
        });
        return cnt;
    }

    u64 countw(Integer w_id) {
        u8 k[sizeof(Integer)];
        fold(k, w_id);
        u64 cnt = 0;
        u8 kk[Record::maxFoldLength()];
        tree.scanAsc({k, sizeof(Integer)}, [&](BTreeNode& node, unsigned slot) {
            memcpy(kk, node.getPrefix(), node.prefixLen);
            memcpy(kk + node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
            if (memcmp(k, kk, sizeof(Integer)) != 0) return false;
            cnt++;
            return true;
        });
        return cnt;
    }
};

template <class Fn>
void parallel_for(uint64_t begin, uint64_t end, uint64_t nthreads, Fn fn) {
    std::vector<std::thread> threads;
    uint64_t n = end - begin;
    if (n < nthreads) nthreads = n;
    uint64_t perThread = n / nthreads;
    for (unsigned i = 0; i < nthreads; i++) {
        threads.emplace_back([&, i]() {
            uint64_t b = (perThread * i) + begin;
            uint64_t e = (i == (nthreads - 1)) ? end : (b + perThread);
            fn(i, b, e);
        });
    }
    for (auto& t : threads) t.join();
}

#endif  // CALICO_HPP