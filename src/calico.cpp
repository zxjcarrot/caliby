#include "calico.hpp"
#include "catalog.hpp"
#include "logging.hpp"

#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// --- Intrinsics Headers ---
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "perf_util.hpp"
#include "tpcc/ZipfianGenerator.hpp"

// Definitions for thread-local variables
__thread uint16_t workerThreadId = 0;
__thread int32_t tpcchistorycounter = 0;
__thread PID lastAllocatedGlobalPid = 0;  // Track last allocated global PID for multi-index mmap mode

using namespace std;
using namespace tpcc;

// Definition for global BufferManager instance
BufferManager* bm_ptr = nullptr;

// Flag to track if the system has been closed (index arrays unregistered)
bool system_closed = false;

// Definition for PageState static member
bool PageState::packedMode = false;

// Definition for LibaioInterface static member
const u64 LibaioInterface::maxIOs;

__attribute__((target("sse4.2"))) uint64_t hash64(uint64_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;
    uint64_t crc = _mm_crc32_u64(seed, key);
    return crc * ((k << 32) + 1);
}

uint64_t rdtsc() {
    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// exmap helper function
static int exmapAction(int exmapfd, exmap_opcode op, u16 len) {
    struct exmap_action_params params_free = {
        .interface = workerThreadId,
        .iov_len = len,
        .opcode = (u16)op,
    };
    return ioctl(exmapfd, EXMAP_IOCTL_ACTION, &params_free);
}

// Calculate minimal number of bits needed to represent n
static u32 bitsNeeded(u64 n) {
    if (n == 0) return 1;
    return 64 - __builtin_clzll(n - 1);
}

//=============================================================================
// IndexCatalog Implementation
//=============================================================================

void IndexCatalog::registerIndex(u32 index_id, u64 max_pages, int file_fd) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    auto it = entries.find(index_id);
    if (it != entries.end()) {
        // Already registered
        return;
    }
    
    IndexEntry entry(index_id, max_pages);
    entry.file_fd = file_fd;
    entries[index_id] = std::move(entry);
    
    CALIBY_LOG_DEBUG("IndexCatalog", "Registered index ", index_id, 
                     " with max_pages=", max_pages);
}

void IndexCatalog::updateAllocCount(u32 index_id, u64 count) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    auto it = entries.find(index_id);
    if (it != entries.end()) {
        u64 current = it->second.alloc_count.load(std::memory_order_relaxed);
        if (count > current) {
            it->second.alloc_count.store(count, std::memory_order_release);
        }
    }
}

u64 IndexCatalog::getAllocCount(u32 index_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    auto it = entries.find(index_id);
    if (it != entries.end()) {
        return it->second.alloc_count.load(std::memory_order_acquire);
    }
    return 0;  // Default starting value
}

int IndexCatalog::getFileFd(u32 index_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    auto it = entries.find(index_id);
    if (it != entries.end()) {
        return it->second.file_fd;
    }
    return -1;  // Not found
}

void IndexCatalog::persist() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    
    std::ofstream ofs(catalog_file_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        CALIBY_LOG_ERROR("IndexCatalog", "Failed to open catalog file for writing: ", 
                         catalog_file_path);
        return;
    }
    
    // Write number of entries
    u32 num_entries = entries.size();
    ofs.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    
    // Write each entry
    for (const auto& pair : entries) {
        const IndexEntry& entry = pair.second;
        u64 alloc_count_value = entry.alloc_count.load(std::memory_order_acquire);
        
        ofs.write(reinterpret_cast<const char*>(&entry.index_id), sizeof(entry.index_id));
        ofs.write(reinterpret_cast<const char*>(&alloc_count_value), sizeof(alloc_count_value));
        ofs.write(reinterpret_cast<const char*>(&entry.max_pages), sizeof(entry.max_pages));
        ofs.write(reinterpret_cast<const char*>(&entry.file_fd), sizeof(entry.file_fd));
    }
    
    ofs.close();
    CALIBY_LOG_DEBUG("IndexCatalog", "Persisted ", num_entries, " entries to ", 
                     catalog_file_path);
}

void IndexCatalog::load() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    std::ifstream ifs(catalog_file_path, std::ios::binary);
    if (!ifs) {
        CALIBY_LOG_INFO("IndexCatalog", "Catalog file not found, starting fresh: ", 
                        catalog_file_path);
        return;
    }
    
    // Read number of entries
    u32 num_entries = 0;
    ifs.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
    
    // Read each entry
    for (u32 i = 0; i < num_entries; i++) {
        IndexEntry entry;
        u64 alloc_count_value;
        
        ifs.read(reinterpret_cast<char*>(&entry.index_id), sizeof(entry.index_id));
        ifs.read(reinterpret_cast<char*>(&alloc_count_value), sizeof(alloc_count_value));
        ifs.read(reinterpret_cast<char*>(&entry.max_pages), sizeof(entry.max_pages));
        ifs.read(reinterpret_cast<char*>(&entry.file_fd), sizeof(entry.file_fd));
        
        // Reset file_fd since file descriptors are not portable across processes
        // The fd will be re-acquired when the index is actually used
        entry.file_fd = -1;
        
        entry.alloc_count.store(alloc_count_value, std::memory_order_release);
        entries[entry.index_id] = std::move(entry);
    }
    
    ifs.close();
    CALIBY_LOG_DEBUG("IndexCatalog", "Loaded ", num_entries, " entries from ", 
                     catalog_file_path);
}

void IndexCatalog::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    entries.clear();
    CALIBY_LOG_DEBUG("IndexCatalog", "Cleared all entries");
}

//=============================================================================
// IndexTranslationArray Implementation
//=============================================================================

IndexTranslationArray::IndexTranslationArray(u32 indexId, u64 maxPages, u64 initialAllocCount, int fd)
    : capacity(maxPages), file_fd(fd), index_id(indexId), 
      allocCount(indexId == 0 ? initialAllocCount : (initialAllocCount == 0 ? 1 : initialAllocCount)) {
    
    // Ensure minimum initial capacity
    u64 initialCapacity = std::max(maxPages, MIN_INITIAL_CAPACITY);
    capacity.store(initialCapacity, std::memory_order_relaxed);
    
    // Allocate page state array using mmap (NOT huge pages for fine-grained hole punching)
    size_t arraySize = initialCapacity * sizeof(PageState);
    pageStates = (PageState*)mmap(nullptr, arraySize, 
                                   PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pageStates == MAP_FAILED) {
        throw std::runtime_error("IndexTranslationArray: Failed to mmap page state array");
    }
    
    // Calculate number of OS page groups for ref counting
    numRefCountGroups = (arraySize + TRANSLATION_OS_PAGE_SIZE - 1) / TRANSLATION_OS_PAGE_SIZE;
    
    // Allocate reference counts array
    refCounts = new std::atomic<u32>[numRefCountGroups]();
    
    CALIBY_LOG_DEBUG("IndexTranslationArray", "Created for index ", indexId,
                     " with initial capacity=", initialCapacity, " pages (growable)",
                     " (", numRefCountGroups, " ref count groups)");
}

bool IndexTranslationArray::ensureCapacity(u64 minCapacity) {
    // Fast path: check if capacity is already sufficient (lock-free read)
    u64 currentCapacity = capacity.load(std::memory_order_acquire);
    if (currentCapacity >= minCapacity) {
        return true;
    }
    
    // Slow path: need to grow - acquire lock
    std::lock_guard<std::mutex> lock(growMutex);
    
    // Double-check after acquiring lock (another thread may have grown)
    currentCapacity = capacity.load(std::memory_order_acquire);
    if (currentCapacity >= minCapacity) {
        return true;
    }
    
    // Calculate new capacity (at least double, or enough for minCapacity)
    u64 newCapacity = currentCapacity;
    while (newCapacity < minCapacity) {
        newCapacity *= GROWTH_FACTOR;
    }
    
    // Use mremap to grow the array in-place (Linux-specific, very efficient)
    size_t oldSize = currentCapacity * sizeof(PageState);
    size_t newSize = newCapacity * sizeof(PageState);
    
    void* newPageStates = mremap(pageStates, oldSize, newSize, MREMAP_MAYMOVE);
    if (newPageStates == MAP_FAILED) {
        CALIBY_LOG_ERROR("IndexTranslationArray", "mremap failed for index ", index_id,
                         " trying to grow from ", currentCapacity, " to ", newCapacity,
                         " pages, errno: ", errno);
        return false;
    }
    
    pageStates = (PageState*)newPageStates;
    
    // Grow reference counts array
    u64 newNumRefCountGroups = (newSize + TRANSLATION_OS_PAGE_SIZE - 1) / TRANSLATION_OS_PAGE_SIZE;
    if (newNumRefCountGroups > numRefCountGroups) {
        std::atomic<u32>* newRefCounts = new std::atomic<u32>[newNumRefCountGroups]();
        // Copy old ref counts
        for (u64 i = 0; i < numRefCountGroups; i++) {
            newRefCounts[i].store(refCounts[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        delete[] refCounts;
        refCounts = newRefCounts;
        numRefCountGroups = newNumRefCountGroups;
    }
    
    // Update capacity (atomic store with release semantics so other threads see new capacity)
    capacity.store(newCapacity, std::memory_order_release);
    
    CALIBY_LOG_DEBUG("IndexTranslationArray", "Grew index ", index_id,
                     " from ", currentCapacity, " to ", newCapacity, " pages");
    
    return true;
}

IndexTranslationArray::~IndexTranslationArray() {
    // Print ref count histogram before cleanup
    // if (refCounts != nullptr) {
    //     std::unordered_map<u32, u64> histogram;
    //     for (u64 i = 0; i < numRefCountGroups; i++) {
    //         u32 count = refCounts[i].load(std::memory_order_relaxed) & REF_COUNT_MASK;
    //         histogram[count]++;
    //     }
    //     std::cerr << "[IndexTranslationArray] Ref Count Histogram for index " << index_id << ":" << std::endl;
    //     std::vector<std::pair<u32, u64>> sorted_histogram(histogram.begin(), histogram.end());
    //     std::sort(sorted_histogram.begin(), sorted_histogram.end());
    //     for (const auto& entry : sorted_histogram) {
    //         std::cerr << "  Ref Count " << entry.first << ": " << entry.second << " groups" << std::endl;
    //     }
    // }
    
    if (pageStates != nullptr && pageStates != MAP_FAILED) {
        size_t arraySize = capacity.load(std::memory_order_relaxed) * sizeof(PageState);
        munmap(pageStates, arraySize);
    }
    delete[] refCounts;
    
    //std::cerr << "[IndexTranslationArray] Destroyed for index " << index_id << std::endl;
}

void IndexTranslationArray::incrementRefCount(u64 localPageId) {
    u64 group = getRefCountGroup(localPageId);
    if (group < numRefCountGroups) {
        // Atomically increment reference count with lock bit handling
        while (true) {
            u32 oldVal = refCounts[group].load(std::memory_order_acquire);
            
            // Check if locked (highest bit set)
            if (oldVal & REF_COUNT_LOCK_BIT) {
                _mm_pause();  // Spin-wait during hole-punching
                continue;
            }
            
            // Try to increment count (lower 31 bits)
            u32 count = oldVal & REF_COUNT_MASK;
            u32 newVal = (count + 1) & REF_COUNT_MASK;
            
            if (refCounts[group].compare_exchange_weak(oldVal, newVal, 
                                                        std::memory_order_release,
                                                        std::memory_order_acquire)) {
                break;  // Successfully incremented
            }
        }
    }
}

void IndexTranslationArray::decrementRefCount(u64 localPageId, std::atomic<u64>& holePunchCounter) {
    u64 group = getRefCountGroup(localPageId);
    if (group >= numRefCountGroups) return;
    // Atomically decrement reference count with lock bit handling
    while (true) {
        u32 oldVal = refCounts[group].load(std::memory_order_acquire);
        
        // Check if locked (highest bit set)
        if (oldVal & REF_COUNT_LOCK_BIT) {
            _mm_pause();
            continue;
        }
        
        // Try to acquire lock first
        u32 lockedVal = oldVal | REF_COUNT_LOCK_BIT;
        if (refCounts[group].compare_exchange_weak(oldVal, lockedVal,
                                                    std::memory_order_acquire,
                                                    std::memory_order_acquire)) {
            // We have the lock
            u32 count = oldVal & REF_COUNT_MASK;
            u32 oldCount = count;
            if (count > 0) {
                count--;
            }
            
            if (count == 0) {
                // Perform hole-punching via madvise on this index's translation array
                void* osPageStart = reinterpret_cast<void*>(
                    reinterpret_cast<uintptr_t>(pageStates) + 
                    (group * TRANSLATION_OS_PAGE_SIZE)
                );
                
                int result = madvise(osPageStart, TRANSLATION_OS_PAGE_SIZE, MADV_DONTNEED);
                if (result == 0) {
                    holePunchCounter.fetch_add(1, std::memory_order_relaxed);
                    } else {
                    CALIBY_LOG_WARN("IndexTranslationArray", "madvise MADV_DONTNEED failed for localPageId ", 
                                    localPageId, " group ", group, " errno: ", errno);
                }
            }
            
            // Release lock and store new count
            refCounts[group].store(count, std::memory_order_release);
            return;
        }
    }
}

//=============================================================================
// TwoLevelPageStateArray Implementation (Multi-Index Support)
//=============================================================================

TwoLevelPageStateArray::TwoLevelPageStateArray(u64 defaultIndexCapacity)
    : numIndexSlots(MAX_INDEXES), holePunchCounterPtr(nullptr), defaultArray(nullptr) {
    
    // Allocate top-level array of atomic pointers
    size_t topArraySize = numIndexSlots * sizeof(std::atomic<IndexTranslationArray*>);
    indexArrays = new std::atomic<IndexTranslationArray*>[numIndexSlots]();
    
    // Initialize all pointers to nullptr
    for (u32 i = 0; i < numIndexSlots; i++) {
        indexArrays[i].store(nullptr, std::memory_order_relaxed);
    }
    
    // If default capacity specified, create index 0 for backward compatibility
    if (defaultIndexCapacity > 0) {
        defaultArray = new IndexTranslationArray(0, defaultIndexCapacity);
        indexArrays[0].store(defaultArray, std::memory_order_release);
    }
    
    CALIBY_LOG_DEBUG("TwoLevelPageStateArray", "Created multi-index translation array",
                     " with ", numIndexSlots, " index slots",
                     (defaultIndexCapacity > 0 ? std::string(", default index capacity=") + std::to_string(defaultIndexCapacity) : ""));
}

TwoLevelPageStateArray::~TwoLevelPageStateArray() {
    // Delete all per-index arrays
    for (u32 i = 0; i < numIndexSlots; i++) {
        IndexTranslationArray* arr = indexArrays[i].load(std::memory_order_acquire);
        if (arr != nullptr) {
            delete arr;
        }
    }
    delete[] indexArrays;
    
    CALIBY_LOG_DEBUG("TwoLevelPageStateArray", "Destroyed multi-index translation array");
}

void TwoLevelPageStateArray::registerIndex(u32 indexId, u64 maxPages, u64 initialAllocCount, int fileFd) {
    if (indexId >= numIndexSlots) {
        throw std::out_of_range("TwoLevelPageStateArray::registerIndex: indexId out of range");
    }
    
    std::unique_lock<std::shared_mutex> lock(indexMutex);
    
    // Check if already registered
    IndexTranslationArray* existing = indexArrays[indexId].load(std::memory_order_acquire);
    if (existing != nullptr) {
        throw std::runtime_error("TwoLevelPageStateArray::registerIndex: index already registered");
    }
    
    // Create new per-index translation array
    IndexTranslationArray* newArray = new IndexTranslationArray(indexId, maxPages, initialAllocCount, fileFd);
    indexArrays[indexId].store(newArray, std::memory_order_release);
    
    // Update default array pointer if this is index 0
    if (indexId == 0) {
        defaultArray = newArray;
    }
    
    CALIBY_LOG_DEBUG("TwoLevelPageStateArray", "Registered index ", indexId, 
                     " with capacity ", maxPages);
}

void TwoLevelPageStateArray::unregisterIndex(u32 indexId) {
    if (indexId >= numIndexSlots) {
        throw std::out_of_range("TwoLevelPageStateArray::unregisterIndex: indexId out of range");
    }
    
    std::unique_lock<std::shared_mutex> lock(indexMutex);
    
    IndexTranslationArray* arr = indexArrays[indexId].load(std::memory_order_acquire);
    if (arr == nullptr) {
        throw std::runtime_error("TwoLevelPageStateArray::unregisterIndex: index not registered");
    }
    
    // Clear the pointer first
    indexArrays[indexId].store(nullptr, std::memory_order_release);
    
    // Update default array if needed
    if (indexId == 0) {
        defaultArray = nullptr;
    }
    
    // Delete the array
    delete arr;
    
    // Increment generation to invalidate all thread-local caches
    generation.fetch_add(1, std::memory_order_release);
    
    CALIBY_LOG_DEBUG("TwoLevelPageStateArray", "Unregistered index ", indexId);
}

void TwoLevelPageStateArray::unregisterAllNonZero() {
    std::unique_lock<std::shared_mutex> lock(indexMutex);
    
    int unregisteredCount = 0;
    for (u32 i = 1; i < numIndexSlots; ++i) {
        IndexTranslationArray* arr = indexArrays[i].load(std::memory_order_acquire);
        if (arr != nullptr) {
            indexArrays[i].store(nullptr, std::memory_order_release);
            delete arr;
            unregisteredCount++;
        }
    }
    
    if (unregisteredCount > 0) {
        // Increment generation to invalidate all thread-local caches
        generation.fetch_add(1, std::memory_order_release);
        
        CALIBY_LOG_DEBUG("TwoLevelPageStateArray", "Unregistered ", unregisteredCount, 
                         " non-zero indexes");
    }
}

bool TwoLevelPageStateArray::isIndexRegistered(u32 indexId) const {
    if (indexId >= numIndexSlots) {
        return false;
    }
    std::shared_lock<std::shared_mutex> lock(indexMutex);
    return indexArrays[indexId].load(std::memory_order_acquire) != nullptr;
}

IndexTranslationArray* TwoLevelPageStateArray::getIndexArray(u32 indexId) const {
    if (indexId >= numIndexSlots) {
        return nullptr;
    }
    return indexArrays[indexId].load(std::memory_order_acquire);
}

void TwoLevelPageStateArray::incrementRefCount(PID pid) {
    u32 indexId = getIndexId(pid);
    u32 localPageId = getLocalPageId(pid);
    
    IndexTranslationArray* arr = (indexId == 0 && defaultArray) 
                                  ? defaultArray 
                                  : indexArrays[indexId].load(std::memory_order_acquire);
    if (arr) {
        arr->incrementRefCount(localPageId);
    }
}

void TwoLevelPageStateArray::decrementRefCount(PID pid) {
    u32 indexId = getIndexId(pid);
    u32 localPageId = getLocalPageId(pid);
    
    IndexTranslationArray* arr = (indexId == 0 && defaultArray)
                                  ? defaultArray
                                  : indexArrays[indexId].load(std::memory_order_acquire);
    if (arr && holePunchCounterPtr) {
        arr->decrementRefCount(localPageId, *holePunchCounterPtr);
    } else {
        }
}

u64 TwoLevelPageStateArray::getIndexCapacity(u32 indexId) const {
    if (indexId >= numIndexSlots) {
        return 0;
    }
    std::shared_lock<std::shared_mutex> lock(indexMutex);
    IndexTranslationArray* arr = indexArrays[indexId].load(std::memory_order_acquire);
    if (arr == nullptr) {
        return 0;
    }
    return arr->capacity.load(std::memory_order_relaxed);
}

std::vector<std::pair<u32, u64>> TwoLevelPageStateArray::getAllIndexCapacities() const {
    std::vector<std::pair<u32, u64>> result;
    std::shared_lock<std::shared_mutex> lock(indexMutex);
    
    for (u32 i = 0; i < numIndexSlots; ++i) {
        IndexTranslationArray* arr = indexArrays[i].load(std::memory_order_acquire);
        if (arr != nullptr) {
            u64 cap = arr->capacity.load(std::memory_order_relaxed);
            result.emplace_back(i, cap);
        }
    }
    return result;
}

// ThreeLevelPageStateArray implementation
ThreeLevelPageStateArray::ThreeLevelPageStateArray(u64 virtCount) 
    : total_entries(virtCount) {
    
    // Calculate minimal bits needed for total entries
    u32 totalBits = bitsNeeded(virtCount);
    
    // Top bits = total - middle - bottom bits
    u32 usedBits = bottomBits + middleBits;
    topBits = (totalBits > usedBits) ? (totalBits - usedBits) : 0;
    num_top_slots = 1U << topBits;
    num_middle_slots = 1U << middleBits;
    entries_per_slot = 1U << bottomBits;
    
    CALIBY_LOG_DEBUG("ThreeLevelPageStateArray", "virtCount=", virtCount, 
                     " totalBits=", totalBits,
                     " topBits=", topBits, " (", num_top_slots, " slots)",
                     " middleBits=", middleBits, " (", num_middle_slots, " slots)",
                     " bottomBits=", bottomBits, " (", entries_per_slot, " entries/slot)");
    
    // Allocate top-level array
    size_t top_size = num_top_slots * sizeof(PageState**);
    top_level_arrays = (PageState***)allocHuge(top_size);
    
    // Allocate each middle-level array
    size_t middle_size = num_middle_slots * sizeof(PageState*);
    for (u32 i = 0; i < num_top_slots; i++) {
        top_level_arrays[i] = (PageState**)allocHuge(middle_size);
        
        // Allocate each bottom-level array
        size_t bottom_level_size = entries_per_slot * sizeof(PageState);
        for (u32 j = 0; j < num_middle_slots; j++) {
            top_level_arrays[i][j] = (PageState*)allocHuge(bottom_level_size);
            
            // Initialize all entries in this bottom-level array
            for (u32 k = 0; k < entries_per_slot; k++) {
                top_level_arrays[i][j][k].init();
            }
        }
    }
}

ThreeLevelPageStateArray::~ThreeLevelPageStateArray() {
    if (top_level_arrays) {
        size_t bottom_level_size = entries_per_slot * sizeof(PageState);
        size_t middle_size = num_middle_slots * sizeof(PageState*);
        
        for (u32 i = 0; i < num_top_slots; i++) {
            if (top_level_arrays[i]) {
                for (u32 j = 0; j < num_middle_slots; j++) {
                    if (top_level_arrays[i][j]) {
                        munmap(top_level_arrays[i][j], bottom_level_size);
                    }
                }
                munmap(top_level_arrays[i], middle_size);
            }
        }
        size_t top_size = num_top_slots * sizeof(PageState**);
        munmap(top_level_arrays, top_size);
    }
}

// allocate memory using huge pages
void* allocHuge(size_t size) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(p, size, MADV_HUGEPAGE);
    return p;
}

// use when lock is not free
void yield(u64 counter) { 

}

ResidentPageSet::ResidentPageSet(u64 maxCount) : count(next_pow2(maxCount * 1.5)), mask(count - 1), clockPos(0) {
    ht = (Entry*)allocHuge(count * sizeof(Entry));
    memset((void*)ht, 0xFF, count * sizeof(Entry));
}

ResidentPageSet::~ResidentPageSet() { munmap(ht, count * sizeof(u64)); }

u64 ResidentPageSet::next_pow2(u64 x) { return 1 << (64 - __builtin_clzl(x - 1)); }

u64 ResidentPageSet::hash(u64 k) {
    const u64 m = 0xc6a4a7935bd1e995;
    const int r = 47;
    u64 h = 0x8445d61a4e774912 ^ (8 * m);
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

void ResidentPageSet::insert(u64 pid) {
    u64 pos = hash(pid) & mask;
    while (true) {
        u64 curr = ht[pos].pid.load();
        assert(curr != pid);
        if ((curr == empty) || (curr == tombstone))
            if (ht[pos].pid.compare_exchange_strong(curr, pid)) return;

        pos = (pos + 1) & mask;
    }
}

bool ResidentPageSet::remove(u64 pid) {
    u64 pos = hash(pid) & mask;
    while (true) {
        u64 curr = ht[pos].pid.load();
        if (curr == empty) return false;

        if (curr == pid)
            if (ht[pos].pid.compare_exchange_strong(curr, tombstone)) return true;

        pos = (pos + 1) & mask;
    }
}

LibaioInterface::LibaioInterface(int blockfd, BufferManager* bm_ptr) : blockfd(blockfd), bm_ptr(bm_ptr) {
    memset(&ctx, 0, sizeof(io_context_t));
    int ret = io_setup(maxIOs, &ctx);
    if (ret != 0) {
        const char* errName = "UNKNOWN";
        switch (-ret) {
            case EAGAIN: errName = "EAGAIN"; break;
            case EFAULT: errName = "EFAULT"; break;
            case EINVAL: errName = "EINVAL"; break;
            case ENOMEM: errName = "ENOMEM"; break;
            case ENOSYS: errName = "ENOSYS"; break;
        }
        CALIBY_LOG_ERROR("LibaioInterface", "io_setup error: ", ret, " ", errName);
        exit(EXIT_FAILURE);
    }
}

void LibaioInterface::writePages(const vector<PID>& pages) {
    assert(pages.size() <= maxIOs);
    
    // Filter out pages with invalid file descriptors
    vector<PID> validPages;
    vector<u64> validIndices;
    validPages.reserve(pages.size());
    validIndices.reserve(pages.size());
    
    for (u64 i = 0; i < pages.size(); i++) {
        PID pid = pages[i];
        Page* page = bm_ptr->preparePageForWrite(pid);
        
        // Get the correct file descriptor and local page ID for this PID
        PID localPageId;
        int fd = bm_ptr->getFileDescriptorForPID(pid, localPageId);
        
        // Skip pages with invalid file descriptors (may happen during shutdown)
        if (fd < 0) {
            continue;
        }
        
        cbPtr[validPages.size()] = &cb[validPages.size()];
        io_prep_pwrite(cb + validPages.size(), fd, page, pageSize, pageSize * localPageId);
        validPages.push_back(pid);
        validIndices.push_back(i);
    }
    
    if (validPages.empty()) {
        return;  // Nothing to write
    }
    
    // Submit IOs, handling partial submissions
    size_t submitted = 0;
    while (submitted < validPages.size()) {
        int cnt = io_submit(ctx, validPages.size() - submitted, &cbPtr[submitted]);
        if (cnt < 0) {
            CALIBY_LOG_ERROR("LibaioInterface", "io_submit failed: ", cnt, " errno: ", -cnt);
            abort();
        }
        if (cnt == 0) {
            // No progress - this shouldn't happen
            CALIBY_LOG_ERROR("LibaioInterface", "io_submit returned 0, no progress possible");
            abort();
        }
        submitted += cnt;
    }
    
    int cnt = io_getevents(ctx, validPages.size(), validPages.size(), events, nullptr);
    if (cnt != (int)validPages.size()) {
        CALIBY_LOG_ERROR("LibaioInterface", "io_getevents failed: ", cnt, " expected: ", validPages.size(), " errno: ", -cnt);
        exit(EXIT_FAILURE);
    }
}

void LibaioInterface::readPages(const vector<PID>& pages, const vector<Page*>& destinations) {
    assert(pages.size() == destinations.size());
    assert(pages.size() <= maxIOs);

    for (u64 i = 0; i < pages.size(); i++) {
        PID pid = pages[i];
        Page* dest = destinations[i];
        cbPtr[i] = &cb[i];
        
        // Get the correct file descriptor and local page ID for this PID
        PID localPageId;
        int fd = bm_ptr->getFileDescriptorForPID(pid, localPageId);
        
        io_prep_pread(cb + i, fd, dest, pageSize, pageSize * localPageId);
    }

    int cnt = io_submit(ctx, pages.size(), cbPtr);
    if (cnt != (int)pages.size()) {
        CALIBY_LOG_ERROR("LibaioInterface", "io_submit read failed: ", cnt, " expected: ", pages.size(), " errno: ", -cnt);
        exit(EXIT_FAILURE);
    }
    cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
    if (cnt != (int)pages.size()) {
        CALIBY_LOG_ERROR("LibaioInterface", "io_getevents read failed: ", cnt, " expected: ", pages.size(), " errno: ", -cnt);
        exit(EXIT_FAILURE);
    }

    // Check results and handle short reads (new file / beyond EOF)
    for (u64 i = 0; i < pages.size(); i++) {
        long res = events[i].res;
        if (res < static_cast<long>(pageSize)) {
            // Page doesn't exist in file yet (new file or beyond EOF)
            // Initialize as an empty/zeroed page
            memset(destinations[i], 0, pageSize);
            destinations[i]->dirty = true;  // Mark dirty so it gets written
        } else {
            destinations[i]->dirty = false;
        }
        bm_ptr->readCount++;
    }
}

u64 envOr(const char* env, u64 value) {
    if (getenv(env)) return atof(getenv(env));
    return value;
}

float envOrF(const char* env, float value) {
    if (getenv(env)) return atof(getenv(env));
    return value;
}

double envOrDouble(const char* env, double value) {
    if (getenv(env)) return atof(getenv(env));
    return value;
}

BufferManager::BufferManager(unsigned nthreads)
    : virtSize(envOr("VIRTGB", 16) * gb),
      physSize(envOrF("PHYSGB", 4) * gb),
      virtCount(virtSize / pageSize),
      physCount(physSize / pageSize),
      residentSet(physCount),
      pageState2Level(nullptr) {
    numThreads = nthreads;
    assert(virtSize >= physSize);
    const char* path = getenv("BLOCK") ? getenv("BLOCK") : "./heapfile";
    blockfd = open(path, O_RDWR | O_DIRECT | O_CREAT, S_IRUSR | S_IWUSR);
    if (blockfd == -1) {
        CALIBY_LOG_ERROR("BufferManager", "cannot open BLOCK device '", path, "'");
        exit(EXIT_FAILURE);
    }

    useTraditional = envOr("TRADITIONAL", 1);
    useExmap = (!useTraditional) && envOr("EXMAP", 0);
    u64 tradHashSetting = useTraditional ? envOr("TRADHASH", 5) : 5;  // Default to Array2Level (multi-index mode)
    if (!useTraditional)
        hashMode = HashMode::Array2Level;  // Use multi-index mode by default
    else if (tradHashSetting == 1)
        hashMode = HashMode::Unordered;
    else if (tradHashSetting == 2)
        hashMode = HashMode::OpenAddress;
    else if (tradHashSetting == 3)
        hashMode = HashMode::Array;
    else if (tradHashSetting == 4)
        hashMode = HashMode::Lockfree;
    else if (tradHashSetting == 5)
        hashMode = HashMode::Array2Level;
    else if (tradHashSetting == 6)
        hashMode = HashMode::Array3Level;
    else
        hashMode = HashMode::Array1Access; 
    PageState::setPackedMode(hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level);
    virtMem = nullptr;
    frameMem = nullptr;
    bool disableHugePageForFrameMem = envOr("NOHUGEPAGE_FRAME_MEM", 0);
    bool disableHugePageForTranslationArray = envOr("NOHUGEPAGE_TRANSLATION_ARRAY", 1);

    u64 virtAllocSize = virtSize + (1 << 16);  // guard space for optimistic reads

    if (useTraditional) {
        frameMem = static_cast<Page*>(allocHuge(physCount * sizeof(Page)));
        if (frameMem == MAP_FAILED) die("allocHuge frameMem");
        if (disableHugePageForFrameMem) {
            madvise(frameMem, physCount * sizeof(Page), MADV_NOHUGEPAGE);
        }
        partitionFrameSpan =
            std::max<u32>(1, static_cast<u32>((physCount + freePartitionCount - 1) / freePartitionCount));
        for (u32 i = 0; i < freePartitionCount; i++) {
            auto& partition = freePartitions[i];
            partition.lock.clear(std::memory_order_relaxed);
            partition.frames.clear();
            u32 start = i * partitionFrameSpan;
            if (start >= physCount) continue;
            u32 end = std::min<u32>(start + partitionFrameSpan, physCount);
            partition.frames.reserve(end - start);
            // Reserve frame 0 as invalid marker - start from frame 1
            u32 frameStart = (start == 0) ? 1 : start;
            for (u32 frame = frameStart; frame < end; frame++) partition.frames.push_back(frame);
            //if (!partition.frames.empty()) std::random_shuffle(partition.frames.begin(), partition.frames.end());
            std::reverse(partition.frames.begin(), partition.frames.end());
            partition.top = partition.frames.size();
        }
        frameToPidArray.reset(new std::atomic<PID>[physCount]);
        for (u64 i = 0; i < physCount; i++) frameToPidArray[i].store(invalidPID, std::memory_order_relaxed);

        if (hashMode == HashMode::Array || hashMode == HashMode::Array1Access) {
            pid_locks = std::make_unique<pthread_rwlock_t[]>(NUM_LOCK_SHARDS);
            for (u64 i = 0; i < NUM_LOCK_SHARDS; ++i) {
                pthread_rwlock_init(&pid_locks[i], nullptr);
            }
        }

        if (hashMode == HashMode::Array) {
            pidToFrameArray.reset(new std::atomic<u64>[virtCount]);
            if (virtCount * sizeof(std::atomic<u64>) >= 2 * mb)
                madvise(pidToFrameArray.get(), virtCount * sizeof(std::atomic<u64>), MADV_HUGEPAGE);
            for (u64 i = 0; i < virtCount; i++) pidToFrameArray[i].store(invalidFrame, std::memory_order_relaxed);
        } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
            pidToFrameArray.reset();
            pidToFrameHash.reset();
            frameToPidHash.reset();
        } else {
            if (hashMode == HashMode::Unordered) {
                if (numThreads == -1) {
                    pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedUnorderedMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedUnorderedMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::OpenAddress) {
                if (numThreads == -1) {
                    pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64>>();
                    frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID>>();
                } else {
                    pidToFrameHash = std::make_unique<ShardedOpenAddressMap<PID, u64, 1>>();
                    frameToPidHash = std::make_unique<ShardedOpenAddressMap<u64, PID, 1>>();
                }
            } else if (hashMode == HashMode::Lockfree) {
                pidToFrameHash = std::make_unique<LockFreeHashMap<PID, u64>>();
                frameToPidHash = std::make_unique<LockFreeHashMap<u64, PID>>();
            } else {
                die("invalid config for hashmap");
            }
            // 
            pidToFrameHash->reserve(physCount * 1.5);
            frameToPidHash->reserve(physCount);
        }
        // round robin cursors will skip empty partitions automatically
    } else {
        if (useExmap) {
            exmapfd = open("/dev/exmap", O_RDWR);
            if (exmapfd < 0) die("open exmap");

            struct exmap_ioctl_setup buffer;
            buffer.fd = blockfd;
            buffer.max_interfaces = maxWorkerThreads;
            buffer.buffer_size = physCount;
            buffer.flags = 0;
            if (ioctl(exmapfd, EXMAP_IOCTL_SETUP, &buffer) < 0) die("ioctl: exmap_setup");

            for (unsigned i = 0; i < maxWorkerThreads; i++) {
                exmapInterface[i] = (struct exmap_user_interface*)mmap(NULL, pageSize, PROT_READ | PROT_WRITE,
                                                                       MAP_SHARED, exmapfd, EXMAP_OFF_INTERFACE(i));
                if (exmapInterface[i] == MAP_FAILED) die("setup exmapInterface");
            }

            virtMem = (Page*)mmap(NULL, virtAllocSize, PROT_READ | PROT_WRITE, MAP_SHARED, exmapfd, 0);
        } else {
            virtMem = (Page*)mmap(NULL, virtAllocSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (disableHugePageForFrameMem) {
                madvise(virtMem, virtAllocSize, MADV_NOHUGEPAGE);
            } else {
                madvise(virtMem, virtAllocSize, MADV_HUGEPAGE);
            }
        }
        if (virtMem == MAP_FAILED) die("mmap failed");
    }

    // Allocate PageState array (single-level or two-level)
    if (hashMode == HashMode::Array2Level) {
        pageState = nullptr;
        // Create multi-index 2-level array with default index 0 having virtCount capacity
        // for backward compatibility with single-index mode
        pageState2Level = new TwoLevelPageStateArray(virtCount);
        pageState2Level->setHolePunchCounter(&holePunchCount);
        pageState3Level.reset();
    } else if (hashMode == HashMode::Array3Level) {
        pageState = nullptr;
        delete pageState2Level;
        pageState2Level = nullptr;
        pageState3Level = std::make_unique<ThreeLevelPageStateArray>(virtCount);
    } else {
        if (disableHugePageForTranslationArray) {
            pageState = (PageState*)mmap(NULL, virtCount * sizeof(PageState), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (pageState == MAP_FAILED) die("mmap pageState");
            madvise(pageState, virtCount * sizeof(PageState), MADV_NOHUGEPAGE);
        } else {
            pageState = (PageState*)allocHuge(virtCount * sizeof(PageState));
        }
        //for (u64 i = 0; i < virtCount; i++) pageState[i].init();
        delete pageState2Level;
        pageState2Level = nullptr;
        pageState3Level.reset();
        // print pageState address
        CALIBY_LOG_DEBUG("BufferManager", "pageState: ", (void*)pageState, " length ", virtCount * sizeof(PageState));
    }
    // Determine OS page size for translation array based on huge page usage
    // Must align with the actual allocation: 4KB if huge pages disabled, 2MB otherwise
    // Set NOHUGEPAGE_TRANSLATION_ARRAY=1 to disable huge pages and enable fine-grained hole-punching
    translationOSPageSize = (disableHugePageForTranslationArray && useTraditional) ? 4096 : (2 * 1024 * 1024);
    
    // Initialize reference counting for hole-punching (Array1Access mode only)
    translationRefCounts = nullptr;
    if (hashMode == HashMode::Array1Access && useTraditional) {
        // Each OS page holds translationOSPageSize/sizeof(PageState) translation entries
        u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
        numRefCountGroups = (virtCount + entriesPerOSPage - 1) / entriesPerOSPage;
        
        // Lazily allocate reference counts using mmap with MAP_PRIVATE | MAP_ANONYMOUS
        // This ensures physical memory is only allocated on first write (zero-page COW)
        size_t refCountSize = numRefCountGroups * sizeof(std::atomic<u32>);
        translationRefCounts = (std::atomic<u32>*)mmap(NULL, refCountSize, PROT_READ | PROT_WRITE,
                                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (translationRefCounts == MAP_FAILED) {
            die("mmap translationRefCounts failed");
        }
        // No need to initialize - mmap with MAP_ANONYMOUS gives zero-filled pages
    }

    physUsedCount = 0;
    allocCount = 0;  // Start from 0; PID 0 will be allocated first as global metadata
    readCount = 0;
    writeCount = 0;
    holePunchCount = 0;
    batch = envOr("BATCH", 64);
    
    // Initialize Index Catalog
    const char* catalog_path = getenv("CATALOG") ? getenv("CATALOG") : "./catalog.dat";
    indexCatalog = std::make_unique<IndexCatalog>(catalog_path);
    indexCatalog->load();  // Load persisted catalog
    
    // Pre-allocate PID 0 as global metadata page for index recovery
    // This ensures GuardX<MetaDataPage>(0) can access it without hanging
    Page* metaPage = allocPage(nullptr);
    assert(toPID(metaPage) == 0);  // First allocation must be PID 0
    // Must unlock the page since allocPage leaves it locked (normally AllocGuard handles this)
    unfixX(0);

    std::string translation_specialization = "general";
    #ifdef CALICO_SPECIALIZATION_CALICO
    translation_specialization = "calico";
    #elif defined(CALICO_SPECIALIZATION_HASH)
    translation_specialization = "hash";
    #else
    #endif
    CALIBY_LOG_INFO("BufferManager", "Initialized: blk:", path, " virtgb:", virtSize / gb, 
                    " physgb:", (float)physSize / gb, " traditional:", useTraditional, 
                    " mmap_os_pagecache:", useMmapOSPageacche, " trad_hash:", static_cast<int>(hashMode),
                    " exmap:", useExmap, " hugepage:", (disableHugePageForFrameMem == 0),
                    " hugepage_translation:", (disableHugePageForTranslationArray == 0),
                    " num_threads:", numThreads, " specialization:", translation_specialization);
}

BufferManager::~BufferManager() {
    try {
        //flushAll();
    } catch (const std::exception& ex) {
        CALIBY_LOG_ERROR("BufferManager", "~BufferManager flush error: ", ex.what());
    } catch (...) {
        CALIBY_LOG_ERROR("BufferManager", "~BufferManager encountered an unknown error during flush.");
    }
    
    // Cleanup lazily allocated reference counts
    if (translationRefCounts != nullptr) {
        // print ref count stats using a histogram style
        // dont use numRefCountGroups as there could be unallocated
        // use allocCount as the maximum page id allocated to estimated the number of groups
        u64 estimatedNumRefCountGroups = (allocCount + (translationOSPageSize / sizeof(PageState)) - 1) / (translationOSPageSize / sizeof(PageState));
        std::unordered_map<u32, u64> histogram;
        for (u64 i = 0; i < estimatedNumRefCountGroups; i++)
        {
            u32 count = translationRefCounts[i].load(std::memory_order_relaxed);
            histogram[count]++;
        }
        if (caliby::is_log_enabled(caliby::LogLevel::DEBUG)) {
            CALIBY_LOG_DEBUG("BufferManager", "Translation Reference Count Histogram:");
            // sort the histogram by <ref count, num groups> and print them
            std::vector<std::pair<u32, u64>> sorted_histogram(histogram.begin(), histogram.end());
            std::sort(sorted_histogram.begin(), sorted_histogram.end());
            for (const auto& entry : sorted_histogram)
            {
                CALIBY_LOG_DEBUG("BufferManager", "  Ref Count ", entry.first, ": ", entry.second, " groups");
            }
        }

        size_t refCountSize = numRefCountGroups * sizeof(std::atomic<u32>);
        munmap(translationRefCounts, refCountSize);
        translationRefCounts = nullptr;
    }

    // print hole punch count
    CALIBY_LOG_DEBUG("BufferManager", "Total hole punches (madvise count): ", holePunchCount);
    
    // Persist catalog before shutdown
    if (indexCatalog) {
        indexCatalog->persist();
    }
}

IOInterface& BufferManager::getIOInterface() {
    static bool useIOUring = envOr("USE_IOURING", 0);
    // if (useIOUring) {
    //     thread_local IOUringInterface instance(blockfd, this);
    //     return instance;
    // } else {
    //     thread_local LibaioInterface instance(blockfd, this);
    //     return instance;
    // }
    thread_local LibaioInterface instance(blockfd, this);
    return instance;
}

PIDAllocator* BufferManager::getOrCreateAllocatorForIndex(u32 index_id, u64 max_pages) {
    // Index 0 uses simple global allocation without per-index encoding
    // Return nullptr to signal using allocPage() instead of allocPageForIndex()
    if (index_id == 0) {
        return nullptr;
    }
    
    // First check if allocator exists
    {
        std::shared_lock<std::shared_mutex> lock(catalogMutex);
        auto it = perIndexAllocators.find(index_id);
        if (it != perIndexAllocators.end()) {
            return it->second.get();
        }
    }
    
    // Create new allocator
    std::unique_lock<std::shared_mutex> lock(catalogMutex);
    
    // Double-check after acquiring write lock
    auto it = perIndexAllocators.find(index_id);
    if (it != perIndexAllocators.end()) {
        return it->second.get();
    }
    
    // Register index in catalog if not already registered
    if (!indexCatalog) {
        throw std::runtime_error("IndexCatalog not initialized");
    }
    
    indexCatalog->registerIndex(index_id, max_pages);
    
    // Initialize from catalog
    u64 saved_alloc_count = indexCatalog->getAllocCount(index_id);
    
    // Get file descriptor from catalog (or fall back to global blockfd)
    int index_fd = indexCatalog->getFileFd(index_id);
    if (index_fd < 0) {
        index_fd = blockfd;  // Fall back to global heapfile if no per-index file
    }
    
    // Register index in the translation array (Array2Level mode)
    if (hashMode == HashMode::Array2Level && pageState2Level) {
        if (!pageState2Level->isIndexRegistered(index_id)) {
            pageState2Level->registerIndex(index_id, max_pages, saved_alloc_count, index_fd);
        }
    }
    
    // Create new allocator
    auto allocator = std::make_unique<PIDAllocator>(index_id);
    
    allocator->next.store(saved_alloc_count, std::memory_order_release);
    allocator->end.store(saved_alloc_count, std::memory_order_release);
    
    PIDAllocator* ptr = allocator.get();
    perIndexAllocators[index_id] = std::move(allocator);
    
    return ptr;
}

int BufferManager::getFileDescriptorForPID(PID pid, PID& localPageId) {
    // In Array2Level mode, extract index_id and get the corresponding file descriptor
    if (hashMode == HashMode::Array2Level && pageState2Level) {
        u32 indexId = pageState2Level->getIndexId(pid);
        localPageId = pageState2Level->getLocalPageId(pid);
        
        IndexTranslationArray* arr = pageState2Level->getIndexArray(indexId);
        if (arr && arr->file_fd >= 0) {
            return arr->file_fd;
        }
    }
    
    // Fall back to global blockfd for single-index mode or if fd not set
    localPageId = pid;
    return blockfd;
}

bool BufferManager::isValidPtr(void* page) {
    if (useTraditional) return (page >= frameMem) && (page < (frameMem + physCount));
    if (!virtMem) return false;
    return (page >= virtMem) && (page < (virtMem + virtCount + 16));
}

PID BufferManager::toPID(void* page) {
    Page* ptr = reinterpret_cast<Page*>(page);
    if (useTraditional) {
        auto idx = ptr - frameMem;
        assert(idx >= 0 && static_cast<u64>(idx) < physCount);
        if (hashMode == HashMode::Array || hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
            PID pid = frameToPidArray[static_cast<size_t>(idx)].load(std::memory_order_acquire);
            assert(pid != invalidPID);
            return pid;
        } else {
            PID pid;
            bool ok = frameToPidHash->tryGet(static_cast<u64>(idx), pid);
            assert(ok);
            return pid;
        }
    }
    // For mmap mode with multi-index, use the thread-local cached global PID
    // This is set by allocPageForIndex just before toPID is called
    if (hashMode == HashMode::Array2Level && lastAllocatedGlobalPid != 0) {
        PID globalPid = lastAllocatedGlobalPid;
        lastAllocatedGlobalPid = 0;  // Reset after use
        return globalPid;
    }
    return ptr - virtMem;
}

inline Page* BufferManager::residentPtr(PID pid) {
    // if (!useTraditional) {
    //     // For multi-index PIDs in mmap mode, get the stored memory offset
    //     if (hashMode == HashMode::Array2Level && pageState2Level) {
    //         u32 indexId = pageState2Level->getIndexId(pid);
    //         if (indexId > 0) {
    //             // Retrieve the actual memory offset from PageState
    //             PageState& ps = getPageState(pid);
    //             u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
    //             u64 memOffset = PageState::extractFrame(stateAndVersion);
    //             return virtMem + memOffset;
    //         }
    //     }
    //     assert(pid < virtCount);
    //     return virtMem + pid;
    // }
    PageState& ps = getPageState(pid);
    u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
#if defined(CALICO_SPECIALIZATION_CALICO)
    return residentPtrCalico(pid, stateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
    return residentPtrHash(pid, stateAndVersion);
#else
    return residentPtr(pid, stateAndVersion);
#endif
}

Page* BufferManager::residentPtrCalico(PID pid, u64 stateAndVersion) {
    // Specialized for Calico array indexing - extracts frame from stateAndVersion
    u64 frame = PageState::extractFrame(stateAndVersion);
    assert(frame != PageState::packedInvalidFrame);
    assert(frame < physCount);
    return frameMem + static_cast<size_t>(frame);
}

Page* BufferManager::residentPtrHash(PID pid, u64 stateAndVersion) {
    // Specialized for hash table lookup - ignores stateAndVersion
    u64 frame;
    bool ok = pidToFrameHash->tryGet(pid, frame);
    assert(ok);
    assert(frame < physCount);
    return frameMem + static_cast<size_t>(frame);
}

Page* BufferManager::residentPtr(PID pid, u64 stateAndVersion) {
    // For mmap mode with multi-index PIDs, we need special handling
    // if (!useTraditional) {
    //     // Check if this is a multi-index PID (index_id in high 32 bits)
    //     if (hashMode == HashMode::Array2Level && pageState2Level) {
    //         u32 indexId = pageState2Level->getIndexId(pid);
    //         if (indexId > 0) {
    //             // For multi-index, retrieve the memory offset from PageState
    //             u64 frame = PageState::extractFrame(stateAndVersion);
    //             if (frame != PageState::packedInvalidFrame && frame < virtCount) {
    //                 return virtMem + frame;
    //             }
    //             // Fallback: read from PageState if not in stateAndVersion
    //             u64 storedFrame = getPageState(pid).getFrameValue();
    //             if (storedFrame != PageState::packedInvalidFrame && storedFrame < virtCount) {
    //                 return virtMem + storedFrame;
    //             }
    //             // Should not reach here for valid multi-index PIDs
    //             std::cerr << "[residentPtr] ERROR: multi-index pid=" << pid << " has invalid frame" << std::endl;
    //             assert(false);
    //         }
    //     }
    //     // For index 0 or non-Array2Level, use pid directly
    //     assert(pid < virtCount);
    //     return virtMem + pid;
    // }

    // For traditional mode with multi-index PIDs in Array2Level mode, the frame is extracted
    // from stateAndVersion - no virtCount assertion needed since PID may have index_id encoded

    //if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
        //pthread_rwlock_rdlock(&pid_locks[pid_lock_shard(pid)]);
        u64 frame = PageState::extractFrame(stateAndVersion);
        // u64 v = ps.getFrameValue();
        // u64 frame = PageState::extractFrame(v);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
        assert(frame != PageState::packedInvalidFrame);
        assert(frame < physCount);
        return frameMem + static_cast<size_t>(frame);
    //}


    // if (hashMode == HashMode::Array) {
    //     //pthread_rwlock_rdlock(&pid_locks[pid_lock_shard(pid)]);
    //     u64 frame = pidToFrameArray[pid].load(std::memory_order_acquire);
    //     //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
    //     assert(frame != invalidFrame);
    //     assert(frame < physCount);
    //     return frameMem + static_cast<size_t>(frame);
    // }

    // u64 frame;
    // bool ok = pidToFrameHash->tryGet(pid, frame);
    // assert(ok);
    // assert(frame < physCount);
    // return frameMem + static_cast<size_t>(frame);
}

Page* BufferManager::toPtr(PID pid) { return residentPtr(pid); }

Page* BufferManager::preparePageForWrite(PID pid) {
    Page* page = residentPtr(pid);
    page->dirty = false;
    return page;
}

static thread_local u32 freePartitionCursor = []() -> u32 {
    auto h = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return static_cast<u32>(h % BufferManager::freePartitionCount);
}();

u32 BufferManager::popFreeFrame() {
    for (u32 attempt = 0; attempt < freePartitionCount; ++attempt) {
        u32 partition = (freePartitionCursor + attempt) % freePartitionCount;
        auto& part = freePartitions[partition];
        if (part.frames.empty()) continue;

        lockPartition(partition);
        if (part.top > 0) {
            u32 frame = part.frames[--part.top];
            unlockPartition(partition);
            freePartitionCursor = (partition + 1) % freePartitionCount;
            return frame;
        }
        unlockPartition(partition);
    }
    throw std::runtime_error("no free frames available");
}

Page* BufferManager::acquireFrameForPid(PID pid) {
    // static std::atomic<int> callCount{0};
    // if (callCount.fetch_add(1) < 10) {
    //     std::cerr << "[acquireFrameForPid] Called for pid=" << pid << " useTraditional=" << useTraditional 
    //               << " hashMode=" << (int)hashMode << std::endl;
    // }
    
    // For non-traditional (mmap) mode, we need to handle multi-index PIDs specially
    // since the encoded PID cannot be used directly as a virtual memory offset
    // if (!useTraditional) {
    //     // Extract actual memory offset for multi-index PIDs
    //     u64 memOffset = pid;
    //     if (hashMode == HashMode::Array2Level && pageState2Level) {
    //         u32 indexId = pageState2Level->getIndexId(pid);
    //         if (indexId > 0) {
    //             // For multi-index, use global allocCount for actual memory allocation
    //             memOffset = allocCount++;
    //             // Store the mapping in the PageState for later retrieval
    //             getPageState(pid).setFrameValue(static_cast<u32>(memOffset));
    //         }
    //     }
    //     return virtMem + memOffset;
    // }
    u32 frame = popFreeFrame();
    if (hashMode == HashMode::Array) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        pidToFrameArray[pid].store(static_cast<u64>(frame), std::memory_order_release);
        frameToPidArray[frame].store(pid, std::memory_order_release);
    } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level || hashMode == HashMode::Array3Level) {
        
        // Increment reference counter for this OS page group (Array1Access only)
        if (hashMode == HashMode::Array1Access && translationRefCounts) {
            u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
            u64 refCountIdx = pid / entriesPerOSPage;
            
            // Atomically increment reference count with lock bit handling
            while (true) {
                u32 oldVal = translationRefCounts[refCountIdx].load(std::memory_order_acquire);
                
                // Check if locked (highest bit set)
                if (oldVal & REF_COUNT_LOCK_BIT) {
                    _mm_pause();  // Spin-wait during hole-punching
                    continue;
                }
                
                // Try to increment count (lower 31 bits)
                u32 count = oldVal & REF_COUNT_MASK;
                u32 newVal = (count + 1) & REF_COUNT_MASK;
                
                if (translationRefCounts[refCountIdx].compare_exchange_weak(oldVal, newVal, 
                                                                            std::memory_order_release,
                                                                            std::memory_order_acquire)) {
                    break;  // Successfully incremented
                }
            }
        }
        
        // Increment reference counter for Array2Level mode (multi-index translation array)
        if (hashMode == HashMode::Array2Level && pageState2Level) {
            // static std::atomic<int> incCount{0};
            // if (incCount.fetch_add(1) < 10) {
            //     std::cerr << "[acquireFrameForPid] Incrementing ref count for pid=" << pid << std::endl;
            // }
            pageState2Level->incrementRefCount(pid);
        }
        
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        getPageState(pid).setFrameValue(frame);
        frameToPidArray[frame].store(pid, std::memory_order_release);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
        
    } else {
        u64 frameId = static_cast<u64>(frame);
        pidToFrameHash->insertOrAssign(pid, frameId);
        frameToPidHash->insertOrAssign(frameId, pid);
    }
    return frameMem + frame;
}

void BufferManager::releaseFrame(PID pid) {
    if (!useTraditional) return;
    
    // static std::atomic<int> callCount{0};
    // if (callCount.fetch_add(1) < 10) {
    //     std::cerr << "[releaseFrame] Called for pid=" << pid << " hashMode=" << (int)hashMode << std::endl;
    // }
    
    u64 frameId;
    if (hashMode == HashMode::Array) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        frameId = pidToFrameArray[pid].exchange(invalidFrame, std::memory_order_acq_rel);
        assert(frameId != invalidFrame);
        frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
    } else if (hashMode == HashMode::Array1Access || hashMode == HashMode::Array2Level) {
        //pthread_rwlock_wrlock(&pid_locks[pid_lock_shard(pid)]);
        frameId = getPageState(pid).getFrameValue();
        assert(frameId != PageState::packedInvalidFrame);
        getPageState(pid).clearFrameValue();
        frameToPidArray[static_cast<size_t>(frameId)].store(invalidPID, std::memory_order_release);
        //pthread_rwlock_unlock(&pid_locks[pid_lock_shard(pid)]);
        
        // Reference counting and hole-punching for both Array1Access and Array2Level
        if ((hashMode == HashMode::Array1Access && translationRefCounts) || 
            (hashMode == HashMode::Array2Level && pageState2Level)) {
            
            // static std::atomic<int> entryCount{0};
            // if (entryCount.fetch_add(1) < 10) {
            //     std::cerr << "[releaseFrame] Entered ref counting branch: hashMode=" << (int)hashMode 
            //               << " translationRefCounts=" << (void*)translationRefCounts
            //               << " pageState2Level=" << (void*)pageState2Level.get() << std::endl;
            // }
            
            if (hashMode == HashMode::Array1Access && translationRefCounts) {
                // Array1Access: single-level translation array with reference counting
                u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
                u64 refCountIdx = pid / entriesPerOSPage;
                
                // Atomically decrement reference count with lock bit handling
                while (true) {
                    u32 oldVal = translationRefCounts[refCountIdx].load(std::memory_order_acquire);
                    
                    // Check if locked (highest bit set)
                    if (oldVal & REF_COUNT_LOCK_BIT) {
                        _mm_pause();
                        continue;
                    }
                    
                    // Try to acquire lock first
                    u32 lockedVal = oldVal | REF_COUNT_LOCK_BIT;
                    if (translationRefCounts[refCountIdx].compare_exchange_weak(oldVal, lockedVal,
                                                                                std::memory_order_acquire,
                                                                                std::memory_order_acquire)) {
                        // We have the lock.
                        u32 count = oldVal & REF_COUNT_MASK;
                        if (count > 0) {
                            count--;
                        }
                        // must obtain the counter lock first before unlocking
                        getPageState(pid).unlockXEvicted();  // Unlock the page state and set it to 0
                        if (count == 0) {
                             // Perform hole-punching via madvise
                            void* osPageStart = reinterpret_cast<void*>(
                                reinterpret_cast<uintptr_t>(pageState) + 
                                (refCountIdx * entriesPerOSPage * sizeof(PageState))
                            );
                            
                            int result = madvise(osPageStart, translationOSPageSize, MADV_DONTNEED);
                            if (result == 0) {
                                holePunchCount.fetch_add(1, std::memory_order_relaxed);
                            } else {
                                CALIBY_LOG_WARN("BufferManager", "madvise MADV_DONTNEED failed for pid ", pid,
                                                " errno: ", errno);
                            }
                        }
                        
                        // Release lock and store new count
                        translationRefCounts[refCountIdx].store(count, std::memory_order_release);
                        break;
                    }
                }
            } else if (hashMode == HashMode::Array2Level && pageState2Level) {
                // Array2Level: multi-index translation array with per-index reference counting
                u32 indexId = pageState2Level->getIndexId(pid);
                u32 localPageId = pageState2Level->getLocalPageId(pid);
                
                IndexTranslationArray* indexArray = pageState2Level->getIndexArray(indexId);
                if (indexArray) {
                    u64 group = indexArray->getRefCountGroup(localPageId);
                    if (group < indexArray->numRefCountGroups) {
                        // Atomically decrement reference count with lock bit handling
                        while (true) {
                            u32 oldVal = indexArray->refCounts[group].load(std::memory_order_acquire);
                            
                            // Check if locked (highest bit set)
                            if (oldVal & REF_COUNT_LOCK_BIT) {
                                _mm_pause();
                                continue;
                            }
                            
                            // Try to acquire lock first
                            u32 lockedVal = oldVal | REF_COUNT_LOCK_BIT;
                            if (indexArray->refCounts[group].compare_exchange_weak(oldVal, lockedVal,
                                                                                    std::memory_order_acquire,
                                                                                    std::memory_order_acquire)) {
                                // We have the lock
                                u32 count = oldVal & REF_COUNT_MASK;
                                if (count > 0) {
                                    count--;
                                }
                                
                                // must obtain the counter lock first before unlocking
                                getPageState(pid).unlockXEvicted();  // Unlock the page state and set it to 0
                                
                                if (count == 0) {
                                    // Perform hole-punching via madvise on this index's translation array
                                    void* osPageStart = reinterpret_cast<void*>(
                                        reinterpret_cast<uintptr_t>(indexArray->pageStates) + 
                                        (group * IndexTranslationArray::TRANSLATION_OS_PAGE_SIZE)
                                    );
                                    
                                    int result = madvise(osPageStart, IndexTranslationArray::TRANSLATION_OS_PAGE_SIZE, MADV_DONTNEED);
                                    if (result == 0) {
                                        holePunchCount.fetch_add(1, std::memory_order_relaxed);
                                    } else {
                                        CALIBY_LOG_WARN("BufferManager", "[Array2Level] madvise MADV_DONTNEED failed for pid ", pid,
                                                        " localPageId=", localPageId, " group=", group,
                                                        " errno: ", errno);
                                    }
                                }
                                
                                // Release lock and store new count
                                indexArray->refCounts[group].store(count, std::memory_order_release);
                                break;
                            }
                        }
                    }
                }
            }
        }
    } else {
        bool ok = pidToFrameHash->erase(pid, frameId);
        assert(ok);
        PID ignored;
        bool ok2 = frameToPidHash->erase(frameId, ignored);
        assert(ok2);
    }

    assert(frameId < physCount);
    u32 frameIdx = static_cast<u32>(frameId);
    u32 partition = partitionForFrame(frameId);
    auto& part = freePartitions[partition];
    lockPartition(partition);
    assert(part.top < part.frames.size());
    part.frames[part.top++] = frameIdx;
    unlockPartition(partition);
}

u32 BufferManager::partitionForFrame(u64 frame) const {
    if (partitionFrameSpan == 0) return 0;
    u32 partition = static_cast<u32>(frame / partitionFrameSpan);
    return (partition >= freePartitionCount) ? (freePartitionCount - 1) : partition;
}

void BufferManager::lockPartition(u32 partition) {
    while (freePartitions[partition].lock.test_and_set(std::memory_order_acquire)) _mm_pause();
}

void BufferManager::unlockPartition(u32 partition) { freePartitions[partition].lock.clear(std::memory_order_release); }

void BufferManager::ensureFreePages() {
    if (physUsedCount >= physCount * 0.95) evict();
}

// allocated new page and fix it
Page* BufferManager::allocPage(PIDAllocator* allocator) {
    physUsedCount++;
    ensureFreePages();
    u64 pid;
    if (allocator) {
        std::lock_guard<std::mutex> g(allocator->lock);
        if (allocator->next >= allocator->end) {
            // make sure the allocated range is aligned to chunkSize
            // ensure allocCount is advanced to the next multiple of chunkSize (512)
            u64 chunkSize = 2048;
            for (;;) {
                u64 start = allocCount.load(std::memory_order_relaxed);
                u64 next_aligned = ((start + 511ULL) & ~511ULL) + chunkSize; // next multiple of 512
                if (allocCount.compare_exchange_weak(start, next_aligned, std::memory_order_acq_rel, std::memory_order_relaxed) == false) continue;
                allocator->next = start;
                allocator->end = next_aligned;
                break;
            }
        }
        pid = allocator->next++;
    } else {
        pid = allocCount++;
    }

    // if (pid >= virtCount) {
    //     cerr << "VIRTGB is too low" << endl;
    //     exit(EXIT_FAILURE);
    // }
    u64 stateAndVersion = getPageState(pid).stateAndVersion;
    bool succ = getPageState(pid).tryLockX(stateAndVersion);
    assert(succ);
    Page* page = acquireFrameForPid(pid);

    if (useExmap) {
        exmapInterface[workerThreadId]->iov[0].page = pid;
        exmapInterface[workerThreadId]->iov[0].len = 1;
        while (exmapAction(exmapfd, EXMAP_OP_ALLOC, 1) < 0) {
            CALIBY_LOG_WARN("BufferManager", "allocPage errno: ", errno, " pid: ", pid, " workerId: ", workerThreadId);
            ensureFreePages();
        }
    }

    // Store global PID for toPID lookup in mmap mode
    lastAllocatedGlobalPid = pid;
    
    // Add to resident set for traditional mode (needed for eviction)
    if (useTraditional) {
        residentSet.insert(pid);
    }

    // DON'T unlock - the AllocGuard will handle that in its destructor
    return page;
}

Page* BufferManager::allocPageForIndex(u32 indexId, PIDAllocator* allocator) {
    // Index 0 must use simple global allocation to avoid PID 0 conflict
    // with the global metadata page
    if (indexId == 0) {
        return allocPage(allocator);
    }
    
    physUsedCount++;
    ensureFreePages();
    
    u64 localPid;
    
    // Get the per-index translation array
    IndexTranslationArray* indexArray = pageState2Level ? pageState2Level->getIndexArray(indexId) : nullptr;
    
    if (!indexArray) {
        // Fallback to global allocCount for index 0 or unregistered indexes
        return allocPage(allocator);
    }
    
    // Allocate from per-index counter
    if (allocator) {
        std::lock_guard<std::mutex> g(allocator->lock);
        if (allocator->next >= allocator->end) {
            u64 chunkSize = 2048;
            for (;;) {
                u64 start = indexArray->allocCount.load(std::memory_order_relaxed);
                u64 next_aligned = ((start + 511ULL) & ~511ULL) + chunkSize;
                if (indexArray->allocCount.compare_exchange_weak(start, next_aligned, std::memory_order_acq_rel, std::memory_order_relaxed) == false) continue;
                allocator->next = start;
                allocator->end = next_aligned;
                break;
            }
        }
        localPid = allocator->next++;
    } else {
        localPid = indexArray->allocCount++;
    }
    
    // Ensure capacity - grow array if needed (truly unbounded growth)
    if (localPid >= indexArray->capacity.load(std::memory_order_acquire)) {
        // Need more capacity - grow the array
        if (!indexArray->ensureCapacity(localPid + 1)) {
            CALIBY_LOG_ERROR("BufferManager", "Index ", indexId, " failed to grow capacity for page ", localPid);
            exit(EXIT_FAILURE);  // Only fail if mremap fails (out of virtual address space)
        }
    }
    
    // Encode as global PID: [index_id (32 bits)][local_page_id (32 bits)]
    u64 globalPid = (static_cast<u64>(indexId) << 32) | (localPid & 0xFFFFFFFFULL);
    
    u64 stateAndVersion = getPageState(globalPid).stateAndVersion;
    
    bool succ = getPageState(globalPid).tryLockX(stateAndVersion);
    assert(succ);
    
    Page* page = acquireFrameForPid(globalPid);
    
    // Store global PID for toPID lookup in mmap mode
    lastAllocatedGlobalPid = globalPid;
    
    // Add to resident set for traditional mode (needed for eviction)
    if (useTraditional) {
        residentSet.insert(globalPid);
    }

    if (useExmap) {
        exmapInterface[workerThreadId]->iov[0].page = globalPid;
        exmapInterface[workerThreadId]->iov[0].len = 1;
        while (exmapAction(exmapfd, EXMAP_OP_ALLOC, 1) < 0) {
            CALIBY_LOG_WARN("BufferManager", "allocPageForIndex errno: ", errno, " pid: ", globalPid, " workerId: ", workerThreadId);
            ensureFreePages();
        }
    }

    // DON'T unlock - the AllocGuard will handle that in its destructor
    return page;
}

void BufferManager::updateAllocCountSnapshot(u64 latest_alloc) {
    while (true) {
        try {
            GuardO<MetaDataPage> meta_guard(metadataPageId);
            if (meta_guard->getAllocCountSnapshot() >= latest_alloc) {
                return;
            }

            GuardX<MetaDataPage> meta_write(std::move(meta_guard));
            if (meta_write->getAllocCountSnapshot() < latest_alloc) {
                meta_write->setAllocCountSnapshot(latest_alloc);
                meta_write->dirty = true;
            }
            return;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
}

Page* BufferManager::handleFault(PID pid) {
    physUsedCount++;
    //ensureFreePages();
    Page* page = acquireFrameForPid(pid);
    readPage(pid, page);
    residentSet.insert(pid);
    return page;
}

Page* BufferManager::fixX(PID pid) {
    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; repeatCounter++) {
        u64 stateAndVersion = ps.stateAndVersion.load();
        u64 state = PageState::getState(stateAndVersion);
        
    if (useMmapOSPageacche) {
            // In mmap+OS page cache mode, pages are never evicted by DB
            // Only need to handle: Locked, Marked, Unlocked states
            if (state == PageState::Marked || state == PageState::Unlocked) {
                if (ps.tryLockX(stateAndVersion)) {
#if defined(CALICO_SPECIALIZATION_CALICO)
                    return residentPtrCalico(pid, stateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
                    return residentPtrHash(pid, stateAndVersion);
#else
                    return residentPtr(pid, stateAndVersion);
#endif
                }
            }
            // For Locked state, just retry
        } else {
            // Original vmcache or traditional mode logic
            switch (state) {
                case PageState::Evicted: {
                    if (ps.tryLockX(stateAndVersion)) {
                        try {
                            Page* page = handleFault(pid);
                            ensureFreePages();
                            return page;
                        } catch (...) {
                            // Unlock page on failure to prevent livelock
                            ps.unlockXEvicted();
                            throw;
                        }
                    }
                    break;
                }
                case PageState::Marked:
                case PageState::Unlocked: {
                    if (ps.tryLockX(stateAndVersion)) {
                        // u64 lockedState = PageState::sameVersion(stateAndVersion, PageState::Locked);
#if defined(CALICO_SPECIALIZATION_CALICO)
                        return residentPtrCalico(pid, stateAndVersion);
#elif defined(CALICO_SPECIALIZATION_HASH)
                        return residentPtrHash(pid, stateAndVersion);
#else
                        return residentPtr(pid, stateAndVersion);
#endif
                    }
                    break;
                }
            }
        }
        yield(repeatCounter);
    }
}

Page* BufferManager::fixS(PID pid) {
    PageState& ps = getPageState(pid);
    for (u64 repeatCounter = 0;; repeatCounter++) {
        u64 stateAndVersion = ps.stateAndVersion;
        u64 state = PageState::getState(stateAndVersion);
        
        if (useMmapOSPageacche) {
            // In mmap+OS page cache mode, pages are never evicted by DB
            // Only need to handle: Locked, Marked, Unlocked, and shared states
            if (state == PageState::Locked) {
                // Just retry
            } else {
                // Try to acquire shared lock (state can be Marked, Unlocked, or already shared)
                if (ps.tryLockS(stateAndVersion)) {
                    u64 s = PageState::getState(stateAndVersion);
                    u64 nextState = (s < PageState::MaxShared) ? (s + 1) : 1;
                    u64 newState = PageState::sameVersion(stateAndVersion, nextState);
#if defined(CALICO_SPECIALIZATION_CALICO)
                    return residentPtrCalico(pid, newState);
#elif defined(CALICO_SPECIALIZATION_HASH)
                    return residentPtrHash(pid, newState);
#else
                    return residentPtr(pid, newState);
#endif
                }
            }
        } else {
            // Original vmcache or traditional mode logic
            switch (state) {
                case PageState::Locked: {
                    break;
                }
                case PageState::Evicted: {
                    if (ps.tryLockX(stateAndVersion)) {
                        try {
                            handleFault(pid);
                            ps.unlockX();
                            ensureFreePages();
                        } catch (...) {
                            // Unlock page on failure to prevent livelock
                            ps.unlockXEvicted();
                            throw;
                        }
                    }
                    break;
                }
                default: {
                    if (ps.tryLockS(stateAndVersion)) {
                        u64 s = PageState::getState(stateAndVersion);
                        u64 nextState = (s < PageState::MaxShared) ? (s + 1) : 1;
                        u64 newState = PageState::sameVersion(stateAndVersion, nextState);
#if defined(CALICO_SPECIALIZATION_CALICO)
                        return residentPtrCalico(pid, newState);
#elif defined(CALICO_SPECIALIZATION_HASH)
                        return residentPtrHash(pid, newState);
#else
                        return residentPtr(pid, newState);
#endif
                    }
                }
            }
        }
        yield(repeatCounter);
    }
}

void BufferManager::unfixS(PID pid) { getPageState(pid).unlockS(); }

void BufferManager::unfixX(PID pid) { 
    getPageState(pid).unlockX();
}

void BufferManager::readPage(PID pid, Page* dest) {
    if (useExmap) {
        for (u64 repeatCounter = 0;; repeatCounter++) {
            int ret = pread(exmapfd, dest, pageSize, workerThreadId);
            if (ret == pageSize) {
                assert(ret == pageSize);
                dest->dirty = false;
                readCount++;
                return;
            }
            CALIBY_LOG_WARN("BufferManager", "readPage errno: ", errno, " pid: ", pid, " workerId: ", workerThreadId);
            ensureFreePages();
        }
    } else {
        // Get the correct file descriptor and local page ID for this PID
        PID localPageId;
        int fd = getFileDescriptorForPID(pid, localPageId);
        
        int ret = pread(fd, dest, pageSize, localPageId * pageSize);
        if (ret < static_cast<int>(pageSize)) {
            // Page doesn't exist in file yet (new file or beyond EOF)
            // Initialize as an empty/zeroed page
            memset(dest, 0, pageSize);
            dest->dirty = true;  // Mark dirty so it gets written
        } else {
            dest->dirty = false;
        }
        readCount++;
    }
}

void BufferManager::flushAll() {
    std::vector<PID> batch;
    batch.reserve(LibaioInterface::maxIOs);
    std::vector<PageState*> locked_states;
    locked_states.reserve(LibaioInterface::maxIOs);
    int flushed_pages = 0;
    auto flush_batch = [&]() {
        if (batch.empty()) return;
        try {
            getIOInterface().writePages(batch);
            writeCount += batch.size();
            flushed_pages += batch.size();
        } catch (...) {
            for (PageState* ps : locked_states) {
                ps->unlockS();
            }
            batch.clear();
            locked_states.clear();
            throw;
        }

        for (PageState* ps : locked_states) {
            ps->unlockS();
        }
        batch.clear();
        locked_states.clear();
    };

    // Lambda to flush pages for a specific global PID
    auto flush_page = [&](PID pid) {
        PageState& ps = getPageState(pid);

        for (u64 repeatCounter = 0;; ++repeatCounter) {
            u64 state_and_version = ps.stateAndVersion.load(std::memory_order_acquire);
            u64 state = PageState::getState(state_and_version);

            if (state == PageState::Evicted) {
                break;
            }
            if (state == PageState::Locked) {
                yield(repeatCounter);
                continue;
            }

            if (ps.tryLockS(state_and_version)) {
#if defined(CALICO_SPECIALIZATION_CALICO)
                Page* page_ptr = residentPtrCalico(pid, state_and_version);
#elif defined(CALICO_SPECIALIZATION_HASH)
                Page* page_ptr = residentPtrHash(pid, state_and_version);
#else
                Page* page_ptr = residentPtr(pid, state_and_version);
#endif
                if (page_ptr->dirty) {
                    batch.push_back(pid);
                    locked_states.push_back(&ps);
                } else {
                    ps.unlockS();
                }
                break;
            }
            yield(repeatCounter);
        }

        if (batch.size() >= LibaioInterface::maxIOs) {
            flush_batch();
        }
    };

    // Flush pages for all registered indices in 2-level mode
    if (pageState2Level) {
        for (u32 indexId = 0; indexId < pageState2Level->numIndexSlots; ++indexId) {
            IndexTranslationArray* arr = pageState2Level->getIndexArray(indexId);
            if (arr) {
                u64 indexAllocCount = arr->allocCount.load(std::memory_order_acquire);
                u64 indexCapacity = arr->capacity.load(std::memory_order_acquire);
                // Use min of allocCount and capacity to avoid overflow
                u64 maxPages = std::min(indexAllocCount, indexCapacity);
                for (u64 localPid = 0; localPid < maxPages; ++localPid) {
                    // Encode global PID: (indexId << 32) | localPid
                    PID globalPid = (static_cast<u64>(indexId) << 32) | localPid;
                    flush_page(globalPid);
                }
            }
        }
    } else {
        // Fallback for non-2level mode: iterate over global allocCount
        u64 max_pid = allocCount.load(std::memory_order_acquire);
        for (PID pid = 0; pid < max_pid; ++pid) {
            flush_page(pid);
        }
    }

    flush_batch();
    CALIBY_LOG_INFO("BufferManager", "Flushed ", flushed_pages, " pages.");
}

void BufferManager::persistIndexCapacities() {
    // Save all index translation array capacities to the IndexCatalog
    // This ensures proper recovery of array sizes on restart
    
    if (hashMode != HashMode::Array2Level || !pageState2Level) {
        return;
    }
    
    auto& catalog = caliby::IndexCatalog::instance();
    if (!catalog.is_initialized()) {
        return;
    }
    
    auto capacities = pageState2Level->getAllIndexCapacities();
    for (const auto& [indexId, capacity] : capacities) {
        // Skip index 0 (default array) if not explicitly created
        if (indexId == 0) continue;
        
        catalog.update_index_alloc_pages(indexId, capacity);
    }
    
    CALIBY_LOG_DEBUG("BufferManager", "Persisted capacities for ", capacities.size(), " indexes");
}

void BufferManager::evict() {
    vector<PID> toEvict;
    toEvict.reserve(batch);
    vector<PID> toWrite;
    toWrite.reserve(batch);

    // 0. find candidates, lock dirty ones in shared mode
    while (toEvict.size() + toWrite.size() < batch) {
        residentSet.iterateClockBatch(batch, [&](PID pid) {
            PageState& ps = getPageState(pid);
            u64 v = ps.stateAndVersion;
            switch (PageState::getState(v)) {
                case PageState::Marked:
                    if (residentPtr(pid)->dirty) {
                        if (ps.tryLockS(v)) toWrite.push_back(pid);
                    } else {
                        toEvict.push_back(pid);
                    }
                    break;
                case PageState::Unlocked:
                    ps.tryMark(v);
                    break;
                default:
                    break;  // skip
            };
        });
    }

    // 1. write dirty pages
    if (toWrite.size() > 0) {
        getIOInterface().writePages(toWrite);
        writeCount += toWrite.size();
    }

    // 2. try to lock clean page candidates
    toEvict.erase(std::remove_if(toEvict.begin(), toEvict.end(),
                                 [&](PID pid) {
                                     PageState& ps = getPageState(pid);
                                     u64 v = ps.stateAndVersion;
                                     return (PageState::getState(v) != PageState::Marked) || !ps.tryLockX(v);
                                 }),
                  toEvict.end());

    // 3. try to upgrade lock for dirty page candidates
    for (auto& pid : toWrite) {
        PageState& ps = getPageState(pid);
        u64 v = ps.stateAndVersion;
        if ((PageState::getState(v) == 1) &&
            ps.stateAndVersion.compare_exchange_weak(v, PageState::sameVersion(v, PageState::Locked)))
            toEvict.push_back(pid); // there were no new readers joined during the write-back, let's evict the page
        else
            ps.unlockS();
    }

    // 4. remove from page table
    if (useExmap) {
        for (u64 i = 0; i < toEvict.size(); i++) {
            exmapInterface[workerThreadId]->iov[i].page = toEvict[i];
            exmapInterface[workerThreadId]->iov[i].len = 1;
        }
        if (exmapAction(exmapfd, EXMAP_OP_FREE, toEvict.size()) < 0) die("ioctl: EXMAP_OP_FREE");
    } else if (!useTraditional) {
        for (u64& pid : toEvict) madvise(virtMem + pid, pageSize, MADV_DONTNEED);
    }

    // 5. remove from hash table and unlock
    // static std::atomic<int> evictLoopCount{0};
    // if (evictLoopCount.fetch_add(1) < 5) {
    //     std::cerr << "[evict] toEvict.size()=" << toEvict.size() << " useTraditional=" << useTraditional << std::endl;
    // }
    for (u64& pid : toEvict) {
        if (useTraditional) {
            releaseFrame(pid);
        }
        bool succ = residentSet.remove(pid);
        assert(succ);
        if (useTraditional && ((hashMode == HashMode::Array1Access && translationRefCounts) || 
                               (hashMode == HashMode::Array2Level && pageState2Level))) {
            // unlockXEvicted has been handled in releaseFrame for hole-punching case
        } else {
            getPageState(pid).unlockXEvicted();
        }
        
    }

    physUsedCount -= toEvict.size();
}

void BufferManager::forceEvictPortion(float portion) {
    if (portion <= 0.0f || portion > 1.0f) {
        CALIBY_LOG_WARN("BufferManager", "forceEvictPortion: Invalid portion ", portion, 
                        ", must be between 0.0 and 1.0");
        return;
    }
    
    if (!useTraditional) {
        CALIBY_LOG_WARN("BufferManager", "forceEvictPortion: Only supported in traditional mode");
        return;
    }
    
    u64 currentUsed = physUsedCount.load(std::memory_order_relaxed);
    u64 targetEvictions = static_cast<u64>(currentUsed * portion);
    if (targetEvictions == 0) {
        CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: No pages to evict (physUsedCount=", 
                         currentUsed, ")");
        return;
    }
    
    CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: Forcing eviction of ", targetEvictions, 
                     " pages (", (portion * 100), "% of ", currentUsed, " resident pages)");
    CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: physCount=", physCount, 
                     ", batch size=", batch);
    
    u64 evicted = 0;
    u64 iterations = 0;
    const u64 maxIterations = (targetEvictions / batch) + 10;  // Safety limit
    
    while (evicted < targetEvictions && iterations < maxIterations) {
        u64 before = physUsedCount.load(std::memory_order_relaxed);
        
        CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: Iteration ", iterations, 
                         ": physUsedCount=", before, ", calling evict()...");
        
        evict();
        
        u64 after = physUsedCount.load(std::memory_order_relaxed);
        u64 evictedThisRound = (before > after) ? (before - after) : 0;
        evicted += evictedThisRound;
        iterations++;
        
        CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: Iteration ", iterations, 
                         " complete: evicted ", evictedThisRound, " pages");
        
        if (evictedThisRound == 0) {
            // No more pages can be evicted
            break;
        }
    }
    
    CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: Evicted ", evicted, " pages in ", 
                     iterations, " iterations");
    CALIBY_LOG_DEBUG("BufferManager", "forceEvictPortion: holePunchCount=", holePunchCount.load());
}

void BufferManager::prefetchPages(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    // Dispatch to appropriate implementation based on hash mode
    if (hashMode == HashMode::Array2Level) {
        prefetchPages2Level(pages, n_pages, offsets_within_pages);
    } else {
        prefetchPagesSingleLevel(pages, n_pages, offsets_within_pages);
    }
}

void BufferManager::prefetchPagesSingleLevel(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    // Optimized prefetch for single-level array modes (Array, Array1Access, etc.)
    std::unique_ptr<vector<PID>> pidsToRead;
    std::unique_ptr<vector<Page*>> destinations;

    // Prefetch state of all pages
    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        //PageState& ps = getPageState(pid);
        PageState& ps = pageState[pid];
        char* tmp = (char*)&ps;
        _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
    }

    // Check which pages are not in memory and need to be loaded
    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        
        if (i + 1 < n_pages) {
            // Prefetch next PageState
            //PageState& next_ps = getPageState(pages[i + 1]);
            PageState& next_ps = pageState[pages[i + 1]];
            char* tmp = (char*)&next_ps;
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
        }

        PageState& ps = pageState[pid];
        u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
        u64 state = PageState::getState(stateAndVersion);

        // Skip pages that are already in memory (not evicted)
        if (state != PageState::Evicted) {
            u64 frame = PageState::extractFrame(stateAndVersion);
            char* tmp = (char*)(frameMem + static_cast<size_t>(frame)) + (offsets_within_pages == nullptr ? 0 : offsets_within_pages[i]);
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
            continue;
        }

        // Lock the page first before allocating frame
        if (ps.tryLockX(stateAndVersion)) {
            ensureFreePages();
            // Try to acquire a frame for this page
            Page* frame = acquireFrameForPid(pid);
            if (frame != nullptr) {
                // Allocate vectors only when we have the first page to read
                if (!pidsToRead) {
                    pidsToRead = std::make_unique<vector<PID>>();
                    destinations = std::make_unique<vector<Page*>>();
                    pidsToRead->reserve(n_pages);
                    destinations->reserve(n_pages);
                }

                pidsToRead->push_back(pid);
                destinations->push_back(frame);
                physUsedCount++;
            } else {
                // Failed to acquire frame, return to evicted state
                ps.unlockXEvicted();
            }
        }
    }

    // Batch read all pages that need to be loaded
    if (pidsToRead && !pidsToRead->empty()) {
        if (pidsToRead->size() >= LibaioInterface::maxIOs) {
            // Split into multiple batches if too many pages
            for (size_t i = 0; i < pidsToRead->size(); i += LibaioInterface::maxIOs) {
                size_t batchSize = std::min(LibaioInterface::maxIOs, pidsToRead->size() - i);
                vector<PID> batch(pidsToRead->begin() + i, pidsToRead->begin() + i + batchSize);
                vector<Page*> batchDest(destinations->begin() + i, destinations->begin() + i + batchSize);
                getIOInterface().readPages(batch, batchDest);
            }
        } else {
            getIOInterface().readPages(*pidsToRead, *destinations);
        }

        // Unlock all successfully loaded pages
        for (PID pid : *pidsToRead) {
            // Add to resident set
            residentSet.insert(pid);
            PageState& ps = getPageState(pid);
            ps.unlockX();
        }
    }
}

void BufferManager::prefetchPages2Level(const PID* pages, int n_pages, const u32* offsets_within_pages) {
    // Optimized for HashMode::Array2Level only, assuming indexId > 0 and all pages from same index
    // Use unique_ptr to defer allocation until we know we have pages to read
    std::unique_ptr<vector<PID>> pidsToRead;
    std::unique_ptr<vector<Page*>> destinations;

    // Get IndexTranslationArray once - all pages assumed to be from same index
    IndexTranslationArray* indexArray = nullptr;
    if (n_pages > 0) {
        u32 indexId = TwoLevelPageStateArray::getIndexId(pages[0]);
        indexArray = pageState2Level->getIndexArray(indexId);
    }
    
    // Early return if no pages to prefetch
    if (n_pages == 0 || indexArray == nullptr) {
        return;
    }
    
    auto pageStates = indexArray->pageStates;

    // prefetch state of all pages
    u64 st = 0;
    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        PageState& ps = pageStates[pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK];
        char * tmp = (char*)&ps;
        _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
    }

    // Check which pages are not in memory and need to be loaded
    for (int i = 0; i < n_pages; i++) {
        PID pid = pages[i];
        
        if (i + 1 < n_pages) {
            // Prefetch next PageState - same index so use same array
            PageState& next_ps = pageStates[pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK];
            char* tmp = (char*)&next_ps;
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
        }

        PageState& ps = pageStates[pid & TwoLevelPageStateArray::LOCAL_PAGE_MASK];
        u64 stateAndVersion = ps.stateAndVersion.load(std::memory_order_acquire);
        u64 state = PageState::getState(stateAndVersion);

        // Skip pages that are already in memory (not evicted)
        if (state != PageState::Evicted) {
            u64 frame = PageState::extractFrame(stateAndVersion);
            char *tmp = (char*)(frameMem + static_cast<size_t>(frame)) + (offsets_within_pages == nullptr ? 0 : offsets_within_pages[i]);
            _mm_prefetch(reinterpret_cast<const char*>(tmp), _MM_HINT_T0);
            continue;
        }

        // Lock the page first before allocating frame
        if (ps.tryLockX(stateAndVersion)) {
            ensureFreePages();
            // Try to acquire a frame for this page
            Page* frame = acquireFrameForPid(pid);
            if (frame != nullptr) {
                // Allocate vectors only when we have the first page to read
                if (!pidsToRead) {
                    pidsToRead = std::make_unique<vector<PID>>();
                    destinations = std::make_unique<vector<Page*>>();
                    pidsToRead->reserve(n_pages);
                    destinations->reserve(n_pages);
                }

                pidsToRead->push_back(pid);
                destinations->push_back(frame);
                physUsedCount++;
            } else {
                // Failed to acquire frame, return to evicted state
                ps.unlockXEvicted();
            }
        }
    }

    // Batch read all pages that need to be loaded
    if (pidsToRead && !pidsToRead->empty()) {
        if (pidsToRead->size() >= LibaioInterface::maxIOs) {
            // Split into multiple batches if too many pages
            for (size_t i = 0; i < pidsToRead->size(); i += LibaioInterface::maxIOs) {
                size_t batchSize = std::min(LibaioInterface::maxIOs, pidsToRead->size() - i);
                vector<PID> batch(pidsToRead->begin() + i, pidsToRead->begin() + i + batchSize);
                vector<Page*> batchDest(destinations->begin() + i, destinations->begin() + i + batchSize);
                getIOInterface().readPages(batch, batchDest);
            }
        } else {
            getIOInterface().readPages(*pidsToRead, *destinations);
        }

        // Unlock all successfully loaded pages
        for (PID pid : *pidsToRead) {
            // Add to resident set
            residentSet.insert(pid);
            PageState& ps = pageState2Level->get(pid, indexArray);
            ps.unlockX();
        }
    }
}

// BTreeNode Method Implementations

static unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }

template <class T>
static T loadUnaligned(void* p) {
    T x;
    memcpy(&x, p, sizeof(T));
    return x;
}

// Get order-preserving head of key (assuming little endian)
static u32 head(u8* key, unsigned keyLen) {
    switch (keyLen) {
        case 0:
            return 0;
        case 1:
            return static_cast<u32>(key[0]) << 24;
        case 2:
            return static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16;
        case 3:
            return (static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16) |
                   (static_cast<u32>(key[2]) << 8);
        default:
            return __builtin_bswap32(loadUnaligned<u32>(key));
    }
}

bool BTreeNode::hasSpaceFor(unsigned keyLen, unsigned payloadLen) {
    return spaceNeeded(keyLen, payloadLen) <= freeSpaceAfterCompaction();
}

PID BTreeNode::getChild(unsigned slotId) { return loadUnaligned<PID>(getPayload(slotId).data()); }

void BTreeNode::makeHint() {
    unsigned dist = count / (hintCount + 1);
    for (unsigned i = 0; i < hintCount; i++) hint[i] = slot[dist * (i + 1)].head;
}

void BTreeNode::updateHint(unsigned slotId) {
    unsigned dist = count / (hintCount + 1);
    unsigned begin = 0;
    if ((count > hintCount * 2 + 1) && (((count - 1) / (hintCount + 1)) == dist) && ((slotId / dist) > 1))
        begin = (slotId / dist) - 1;
    for (unsigned i = begin; i < hintCount; i++) hint[i] = slot[dist * (i + 1)].head;
}

void BTreeNode::searchHint(u32 keyHead, u16& lowerOut, u16& upperOut) {
    if (count > hintCount * 2) {
        u16 dist = upperOut / (hintCount + 1);
        u16 pos, pos2;
        for (pos = 0; pos < hintCount; pos++)
            if (hint[pos] >= keyHead) break;
        for (pos2 = pos; pos2 < hintCount; pos2++)
            if (hint[pos2] != keyHead) break;
        lowerOut = pos * dist;
        if (pos2 < hintCount) upperOut = (pos2 + 1) * dist;
    }
}

u16 BTreeNode::lowerBound(span<u8> skey, bool& foundExactOut) {
    foundExactOut = false;

    // check prefix
    int cmp = memcmp(skey.data(), getPrefix(), min(skey.size(), prefixLen));
    if (cmp < 0)  // key is less than prefix
        return 0;
    if (cmp > 0)  // key is greater than prefix
        return count;
    if (skey.size() < prefixLen)  // key is equal but shorter than prefix
        return 0;
    u8* key = skey.data() + prefixLen;
    unsigned keyLen = skey.size() - prefixLen;

    // check hint
    u16 lower = 0;
    u16 upper = count;
    u32 keyHead = head(key, keyLen);
    searchHint(keyHead, lower, upper);

    // binary search on remaining range
    while (lower < upper) {
        u16 mid = ((upper - lower) / 2) + lower;
        if (keyHead < slot[mid].head) {
            upper = mid;
        } else if (keyHead > slot[mid].head) {
            lower = mid + 1;
        } else {  // head is equal, check full key
            int cmp = memcmp(key, getKey(mid), min(keyLen, slot[mid].keyLen));
            if (cmp < 0) {
                upper = mid;
            } else if (cmp > 0) {
                lower = mid + 1;
            } else {
                if (keyLen < slot[mid].keyLen) {  // key is shorter
                    upper = mid;
                } else if (keyLen > slot[mid].keyLen) {  // key is longer
                    lower = mid + 1;
                } else {
                    foundExactOut = true;
                    return mid;
                }
            }
        }
    }
    return lower;
}

u16 BTreeNode::lowerBound(span<u8> key) {
    bool ignore;
    return lowerBound(key, ignore);
}

void BTreeNode::insertInPage(span<u8> key, span<u8> payload) {
    unsigned needed = spaceNeeded(key.size(), payload.size());
    if (needed > freeSpace()) {
        assert(needed <= freeSpaceAfterCompaction());
        compactify();
    }
    unsigned slotId = lowerBound(key);
    memmove(slot + slotId + 1, slot + slotId, sizeof(Slot) * (count - slotId));
    storeKeyValue(slotId, key, payload);
    count++;
    updateHint(slotId);
}

bool BTreeNode::removeSlot(unsigned slotId) {
    spaceUsed -= slot[slotId].keyLen;
    spaceUsed -= slot[slotId].payloadLen;
    memmove(slot + slotId, slot + slotId + 1, sizeof(Slot) * (count - slotId - 1));
    count--;
    makeHint();
    return true;
}

bool BTreeNode::removeInPage(span<u8> key) {
    bool found;
    unsigned slotId = lowerBound(key, found);
    if (!found) return false;
    return removeSlot(slotId);
}

void BTreeNode::copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src) {
    u64 ofs = offsetof(BTreeNodeHeader, upperInnerNode);
    memcpy(reinterpret_cast<u8*>(dst) + ofs, reinterpret_cast<u8*>(src) + ofs, sizeof(BTreeNode) - ofs);
}

void BTreeNode::compactify() {
    unsigned should = freeSpaceAfterCompaction();
    static_cast<void>(should);
    BTreeNode tmp(isLeaf);
    tmp.setFences(getLowerFence(), getUpperFence());
    copyKeyValueRange(&tmp, 0, 0, count);
    tmp.upperInnerNode = upperInnerNode;
    copyNode(this, &tmp);
    makeHint();
    assert(freeSpace() == should);
}

bool BTreeNode::mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right) {
    if (!isLeaf)
        // TODO: implement inner merge
        return true;

    assert(right->isLeaf);
    assert(parent->isInner());
    BTreeNode tmp(isLeaf);
    tmp.setFences(getLowerFence(), right->getUpperFence());
    unsigned leftGrow = (prefixLen - tmp.prefixLen) * count;
    unsigned rightGrow = (right->prefixLen - tmp.prefixLen) * right->count;
    unsigned spaceUpperBound = spaceUsed + right->spaceUsed +
                               (reinterpret_cast<u8*>(slot + count + right->count) - ptr()) + leftGrow + rightGrow;
    if (spaceUpperBound > pageSize) return false;
    copyKeyValueRange(&tmp, 0, 0, count);
    right->copyKeyValueRange(&tmp, count, 0, right->count);
    PID pid = bm.toPID(this);
    memcpy(parent->getPayload(slotId + 1).data(), &pid, sizeof(PID));
    parent->removeSlot(slotId);
    tmp.makeHint();
    tmp.nextLeafNode = right->nextLeafNode;

    copyNode(this, &tmp);
    return true;
}

void BTreeNode::storeKeyValue(u16 slotId, span<u8> skey, span<u8> payload) {
    // slot
    u8* key = skey.data() + prefixLen;
    unsigned keyLen = skey.size() - prefixLen;
    slot[slotId].head = head(key, keyLen);
    slot[slotId].keyLen = keyLen;
    slot[slotId].payloadLen = payload.size();
    // key
    unsigned space = keyLen + payload.size();
    dataOffset -= space;
    spaceUsed += space;
    slot[slotId].offset = dataOffset;
    assert(getKey(slotId) >= reinterpret_cast<u8*>(&slot[slotId]));
    memcpy(getKey(slotId), key, keyLen);
    memcpy(getPayload(slotId).data(), payload.data(), payload.size());
}

void BTreeNode::copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount) {
    if (prefixLen <= dst->prefixLen) {  // prefix grows
        unsigned diff = dst->prefixLen - prefixLen;
        for (unsigned i = 0; i < srcCount; i++) {
            unsigned newKeyLen = slot[srcSlot + i].keyLen - diff;
            unsigned space = newKeyLen + slot[srcSlot + i].payloadLen;
            dst->dataOffset -= space;
            dst->spaceUsed += space;
            dst->slot[dstSlot + i].offset = dst->dataOffset;
            u8* key = getKey(srcSlot + i) + diff;
            memcpy(dst->getKey(dstSlot + i), key, space);
            dst->slot[dstSlot + i].head = head(key, newKeyLen);
            dst->slot[dstSlot + i].keyLen = newKeyLen;
            dst->slot[dstSlot + i].payloadLen = slot[srcSlot + i].payloadLen;
        }
    } else {
        for (unsigned i = 0; i < srcCount; i++) copyKeyValue(srcSlot + i, dst, dstSlot + i);
    }
    dst->count += srcCount;
    assert((dst->ptr() + dst->dataOffset) >= reinterpret_cast<u8*>(dst->slot + dst->count));
}

void BTreeNode::copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot) {
    unsigned fullLen = slot[srcSlot].keyLen + prefixLen;
    u8 key[fullLen];
    memcpy(key, getPrefix(), prefixLen);
    memcpy(key + prefixLen, getKey(srcSlot), slot[srcSlot].keyLen);
    dst->storeKeyValue(dstSlot, {key, fullLen}, getPayload(srcSlot));
}

void BTreeNode::insertFence(FenceKeySlot& fk, span<u8> key) {
    assert(freeSpace() >= key.size());
    dataOffset -= key.size();
    spaceUsed += key.size();
    fk.offset = dataOffset;
    fk.len = key.size();
    memcpy(ptr() + dataOffset, key.data(), key.size());
}

void BTreeNode::setFences(span<u8> lower, span<u8> upper) {
    insertFence(lowerFence, lower);
    insertFence(upperFence, upper);
    for (prefixLen = 0; (prefixLen < min(lower.size(), upper.size())) && (lower[prefixLen] == upper[prefixLen]);
         prefixLen++);
}

void BTreeNode::splitNode(BTreeNode* parent, unsigned sepSlot, span<u8> sep, PIDAllocator* allocator) {
    assert(sepSlot > 0);
    assert(sepSlot < (pageSize / sizeof(PID)));

    BTreeNode tmp(isLeaf);
    BTreeNode* nodeLeft = &tmp;

    AllocGuard<BTreeNode> newNode(allocator, isLeaf);
    BTreeNode* nodeRight = newNode.ptr;

    nodeLeft->setFences(getLowerFence(), sep);
    nodeRight->setFences(sep, getUpperFence());

    PID leftPID = bm.toPID(this);
    u16 oldParentSlot = parent->lowerBound(sep);
    if (oldParentSlot == parent->count) {
        assert(parent->upperInnerNode == leftPID);
        parent->upperInnerNode = newNode.pid;
    } else {
        assert(parent->getChild(oldParentSlot) == leftPID);
        memcpy(parent->getPayload(oldParentSlot).data(), &newNode.pid, sizeof(PID));
    }
    parent->insertInPage(sep, {reinterpret_cast<u8*>(&leftPID), sizeof(PID)});

    if (isLeaf) {
        copyKeyValueRange(nodeLeft, 0, 0, sepSlot + 1);
        copyKeyValueRange(nodeRight, 0, nodeLeft->count, count - nodeLeft->count);
        nodeLeft->nextLeafNode = newNode.pid;
        nodeRight->nextLeafNode = this->nextLeafNode;
    } else {
        // in inner node split, separator moves to parent (count == 1 + nodeLeft->count + nodeRight->count)
        copyKeyValueRange(nodeLeft, 0, 0, sepSlot);
        copyKeyValueRange(nodeRight, 0, nodeLeft->count + 1, count - nodeLeft->count - 1);
        nodeLeft->upperInnerNode = getChild(nodeLeft->count);
        nodeRight->upperInnerNode = upperInnerNode;
    }
    nodeLeft->makeHint();
    nodeRight->makeHint();
    copyNode(this, nodeLeft);
}

unsigned BTreeNode::commonPrefix(unsigned slotA, unsigned slotB) {
    assert(slotA < count);
    unsigned limit = min(slot[slotA].keyLen, slot[slotB].keyLen);
    u8 *a = getKey(slotA), *b = getKey(slotB);
    unsigned i;
    for (i = 0; i < limit; i++)
        if (a[i] != b[i]) break;
    return i;
}

BTreeNode::SeparatorInfo BTreeNode::findSeparator(bool splitOrdered) {
    assert(count > 1);
    if (isInner()) {
        // inner nodes are split in the middle
        unsigned slotId = count / 2;
        return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[slotId].keyLen), slotId, false};
    }

    // find good separator slot
    unsigned bestPrefixLen, bestSlot;

    if (splitOrdered) {
        bestSlot = count - 2;
    } else if (count > 16) {
        unsigned lower = (count / 2) - (count / 16);
        unsigned upper = (count / 2);

        bestPrefixLen = commonPrefix(lower, 0);
        bestSlot = lower;

        if (bestPrefixLen != commonPrefix(upper - 1, 0))
            for (bestSlot = lower + 1; (bestSlot < upper) && (commonPrefix(bestSlot, 0) == bestPrefixLen); bestSlot++);
    } else {
        bestSlot = (count - 1) / 2;
    }

    // try to truncate separator
    unsigned common = commonPrefix(bestSlot, bestSlot + 1);
    if ((bestSlot + 1 < count) && (slot[bestSlot].keyLen > common) && (slot[bestSlot + 1].keyLen > (common + 1)))
        return SeparatorInfo{prefixLen + common + 1, bestSlot, true};

    return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[bestSlot].keyLen), bestSlot, false};
}

void BTreeNode::getSep(u8* sepKeyOut, SeparatorInfo info) {
    memcpy(sepKeyOut, getPrefix(), prefixLen);
    memcpy(sepKeyOut + prefixLen, getKey(info.slot + info.isTruncated), info.len - prefixLen);
}

PID BTreeNode::lookupInner(span<u8> key) {
    unsigned pos = lowerBound(key);
    if (pos == count) return upperInnerNode;
    return getChild(pos);
}

// BTree Method Implementations

static unsigned btreeslotcounter = 0;

BTree::BTree() : splitOrdered(false) {
    PID rootNodePid;
    {
        AllocGuard<BTreeNode> rootNode(&pidAllocator, true);
        rootNodePid = rootNode.pid;
    }
    
    {
        GuardX<MetaDataPage> page(metadataPageId);
        slotId = btreeslotcounter++;
        page->roots[slotId] = rootNodePid;
        page->dirty = true;
    }
}

// Recovery constructor: reuse existing slotId to recover a persisted BTree
BTree::BTree(unsigned existingSlotId) : slotId(existingSlotId), splitOrdered(false) {
    // Ensure btreeslotcounter stays ahead of recovered slots
    if (existingSlotId >= btreeslotcounter) {
        btreeslotcounter = existingSlotId + 1;
    }
    
    // Verify the slot has a valid root PID in the metadata page
    {
        GuardO<MetaDataPage> meta(metadataPageId);
        PID rootPid = meta->getRoot(slotId);
        if (rootPid == 0) {
            throw std::runtime_error("BTree recovery failed: invalid root PID for slotId " + std::to_string(slotId));
        }
    }
}

BTree::~BTree() {}

GuardO<BTreeNode> BTree::findLeafO(span<u8> key) {
    GuardO<MetaDataPage> meta(metadataPageId);
    GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
    meta.release();

    while (node->isInner()) node = GuardO<BTreeNode>(node->lookupInner(key), node);
    return node;
}

GuardS<BTreeNode> BTree::findLeafS(span<u8> key) {
    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<MetaDataPage> meta(metadataPageId);
            GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
            meta.release();

            while (node->isInner()) node = GuardO<BTreeNode>(node->lookupInner(key), node);

            return GuardS<BTreeNode>(std::move(node));
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

void BTree::trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen) {
    // create new root if necessary
    if (parent.pid == metadataPageId) {
        // parent already holds the lock on metadataPageId, so reuse it
        MetaDataPage* meta_page = reinterpret_cast<MetaDataPage*>(parent.ptr);

        AllocGuard<BTreeNode> newRoot(&pidAllocator, false);
        newRoot->upperInnerNode = node.pid;
        {
            meta_page->roots[slotId] = newRoot.pid;
            meta_page->dirty = true;
        }
        parent = std::move(newRoot);
    }

    // split
    BTreeNode::SeparatorInfo sepInfo = node->findSeparator(splitOrdered.load());
    u8 sepKey[sepInfo.len];
    node->getSep(sepKey, sepInfo);

    if (parent->hasSpaceFor(sepInfo.len, sizeof(PID))) {  // is there enough space in the parent for the separator?
        node->splitNode(parent.ptr, sepInfo.slot, {sepKey, sepInfo.len}, &pidAllocator);
        return;
    }

    // must split parent to make space for separator, restart from root to do this
    node.release();
    parent.release();
    ensureSpace(parent.ptr, {sepKey, sepInfo.len}, sizeof(PID));
}

void BTree::ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen) {
    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<BTreeNode> parent(metadataPageId);
            GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

            while (node->isInner() && (node.ptr != toSplit)) {
                parent = std::move(node);
                node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
            }
            if (node.ptr == toSplit) {
                if (node->hasSpaceFor(key.size(), payloadLen)) return;  // someone else did split concurrently
                GuardX<BTreeNode> parentLocked(std::move(parent));
                GuardX<BTreeNode> nodeLocked(std::move(node));
                trySplit(std::move(nodeLocked), std::move(parentLocked), key, payloadLen);
            }
            return;
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

void BTree::insert(span<u8> key, span<u8> payload) {
    assert((key.size() + payload.size()) <= BTreeNode::maxKVSize);

    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<BTreeNode> parent(metadataPageId);
            GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

            while (node->isInner()) {
                parent = std::move(node);
                node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
            }

            if (node->hasSpaceFor(key.size(), payload.size())) {
                // only lock leaf
                GuardX<BTreeNode> nodeLocked(std::move(node));
                parent.release();
                nodeLocked->insertInPage(key, payload);
                return;  // success
            }

            // lock parent and leaf
            GuardX<BTreeNode> parentLocked(std::move(parent));
            GuardX<BTreeNode> nodeLocked(std::move(node));
            trySplit(std::move(nodeLocked), std::move(parentLocked), key, payload.size());
            // insert hasn't happened, restart from root
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

bool BTree::remove(span<u8> key) {
    for (u64 repeatCounter = 0;; repeatCounter++) {
        try {
            GuardO<BTreeNode> parent(metadataPageId);
            GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

            u16 pos;
            while (node->isInner()) {
                pos = node->lowerBound(key);
                PID nextPage = (pos == node->count) ? node->upperInnerNode : node->getChild(pos);
                parent = std::move(node);
                node = GuardO<BTreeNode>(nextPage, parent);
            }

            bool found;
            unsigned slotId = node->lowerBound(key, found);
            if (!found) return false;

            unsigned sizeEntry = node->slot[slotId].keyLen + node->slot[slotId].payloadLen;
            if ((node->freeSpaceAfterCompaction() + sizeEntry >= BTreeNodeHeader::underFullSize) &&
                (parent.pid != metadataPageId) && (parent->count >= 2) && ((pos + 1) < parent->count)) {
                // underfull
                GuardX<BTreeNode> parentLocked(std::move(parent));
                GuardX<BTreeNode> nodeLocked(std::move(node));
                GuardX<BTreeNode> rightLocked(parentLocked->getChild(pos + 1));
                nodeLocked->removeSlot(slotId);
                if (rightLocked->freeSpaceAfterCompaction() >= (pageSize - BTreeNodeHeader::underFullSize)) {
                    if (nodeLocked->mergeNodes(pos, parentLocked.ptr, rightLocked.ptr)) {
                        // XXX: should reuse page Id
                    }
                }
            } else {
                GuardX<BTreeNode> nodeLocked(std::move(node));
                parent.release();
                nodeLocked->removeSlot(slotId);
            }
            return true;
        } catch (const OLCRestartException&) {
            yield(repeatCounter);
        }
    }
}

u64 BufferManager::countZeroRefCountGroups() {
    if (!translationRefCounts) return 0;
    u64 count = 0;
    u64 entriesPerOSPage = translationOSPageSize / sizeof(PageState);
    u64 currentAllocCount = allocCount.load(std::memory_order_relaxed);
    u64 estimatedNumRefCountGroups = (currentAllocCount + entriesPerOSPage - 1) / entriesPerOSPage;
    
    for (u64 i = 0; i < estimatedNumRefCountGroups; i++) {
        u32 val = translationRefCounts[i].load(std::memory_order_relaxed);
        if ((val & REF_COUNT_MASK) == 0) {
            count++;
        }
    }
    return count;
}





// Buffer configuration (used if env vars not set)
static float config_virtgb = 24.0f;
static float config_physgb = 16.0f;

void set_buffer_config(float virtgb, float physgb) {
    config_virtgb = virtgb;
    config_physgb = physgb;
}

void initialize_system() {
    if (bm_ptr == nullptr) {
        // Set environment variables if they are not set, for default behavior
        // Use config values instead of hardcoded defaults
        setenv("BLOCK", "./heapfile", 0); // Use a different file for python tests
        char virtgb_str[32], physgb_str[32];
        snprintf(virtgb_str, sizeof(virtgb_str), "%.2f", config_virtgb);
        snprintf(physgb_str, sizeof(physgb_str), "%.2f", config_physgb);
        setenv("VIRTGB", virtgb_str, 0);
        setenv("PHYSGB", physgb_str, 0);
        
        bm_ptr = new BufferManager();

        // NOTE: catalog.initialize() is NOT called here - it must be explicitly
        // called via caliby.initialize(data_dir) to specify the data directory

        // Restore allocCount from persisted metadata snapshot if available.
        bool can_attempt_restore = true;
        if (const char* block_path = getenv("BLOCK")) {
            struct stat st {};
            if (stat(block_path, &st) == 0) {
                if (S_ISREG(st.st_mode) && st.st_size < static_cast<off_t>(pageSize)) {
                    can_attempt_restore = false;
                }
            }
        }

        if (can_attempt_restore) {
            for (;;) {
                try {
                    GuardO<MetaDataPage> meta_guard(metadataPageId);
                    u64 stored_alloc = meta_guard->getAllocCountSnapshot();
                    u64 current_alloc = bm_ptr->allocCount.load(std::memory_order_relaxed);
                    if (stored_alloc > current_alloc) {
                        bm_ptr->allocCount.store(stored_alloc, std::memory_order_relaxed);
                        CALIBY_LOG_INFO("Calico", "Restored allocCount from metadata: ", stored_alloc);
                    }
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
        }
    }
}

void flush_system() {
    if (bm_ptr != nullptr) {
        bool can_attempt_update = true;
        if (const char* block_path = getenv("BLOCK")) {
            struct stat st {};
            if (stat(block_path, &st) == 0) {
                if (S_ISREG(st.st_mode) && st.st_size < static_cast<off_t>(pageSize)) {
                    can_attempt_update = false;
                }
            }
        }

        bm_ptr->flushAll();
    }
}

// This function will be called from Python to shut down the system
void shutdown_system() {
    if (bm_ptr != nullptr) {
        flush_system();
        delete bm_ptr;
        bm_ptr = nullptr;
    }
}