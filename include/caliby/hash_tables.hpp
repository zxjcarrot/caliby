#pragma once

#include <unordered_map>
#include <vector>
#include <array>
#include <shared_mutex>
#include <atomic>
#include <sys/mman.h>
#include <mutex>
#include "absl/container/flat_hash_map.h"

// Custom allocator that applies MADV_HUGEPAGE to allocated memory
template <typename T>
class HugePageAllocator {
public:
   using value_type = T;
   
   HugePageAllocator() noexcept = default;
   
   template <typename U>
   HugePageAllocator(const HugePageAllocator<U>&) noexcept {}
   
   T* allocate(size_t n) {
      size_t bytes = n * sizeof(T);
      T* ptr = static_cast<T*>(::operator new(bytes));
      
      // Apply huge page advice if allocation is large enough
      if (bytes >= 2*1024*1024) {  // Only for allocations >= 4KB
         madvise(ptr, bytes, MADV_HUGEPAGE);
      }
      
      return ptr;
   }
   
   void deallocate(T* ptr, size_t n) noexcept {
      ::operator delete(ptr);
   }
   
   template <typename U>
   bool operator==(const HugePageAllocator<U>&) const noexcept {
      return true;
   }
   
   template <typename U>
   bool operator!=(const HugePageAllocator<U>&) const noexcept {
      return false;
   }
};

// Forward declaration of Hasher template
template<typename K>
struct Hasher;

template <typename K, typename V>
class PartitionedMap {
public:
   virtual ~PartitionedMap() = default;
   virtual void reserve(size_t entries) = 0;
   virtual void insertOrAssign(const K& key, const V& value) = 0;
   virtual bool tryGet(const K& key, V& value) const = 0;
   virtual bool erase(const K& key, V& value) = 0;
};

template <typename K, typename V, size_t ShardCount = 64>
class ShardedUnorderedMap : public PartitionedMap<K, V> {
   static_assert((ShardCount & (ShardCount - 1)) == 0, "ShardCount must be a power of two");

   struct Shard {
      std::unordered_map<K, V, Hasher<K>, std::equal_to<K>, HugePageAllocator<std::pair<const K, V>>> map;
      mutable std::shared_mutex mutex;
   };

   std::array<Shard, ShardCount> shards;
   Hasher<K> hasher;

   static constexpr size_t shardMask = ShardCount - 1;

   size_t shardFor(const K& key) const {
      return hasher(key) & shardMask;
   }

public:
   void reserve(size_t entries) override {
      size_t perShard = (entries + ShardCount - 1) / ShardCount;
      for (auto& shard : shards)
         shard.map.reserve(perShard);
   }

   void insertOrAssign(const K& key, const V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      shard.map[key] = value;
   }

   bool tryGet(const K& key, V& value) const override {
      auto& shard = shards[shardFor(key)];
      std::shared_lock lock(shard.mutex);
      auto it = shard.map.find(key);
      if (it == shard.map.end())
         return false;
      value = it->second;
      return true;
   }

   bool erase(const K& key, V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      auto it = shard.map.find(key);
      if (it == shard.map.end())
         return false;
      value = it->second;
      shard.map.erase(it);
      return true;
   }
};

// Single-threaded specialization with no mutex overhead
template <typename K, typename V>
class ShardedUnorderedMap<K, V, 1> : public PartitionedMap<K, V> {
   std::unordered_map<K, V, Hasher<K>> map;

public:
   void reserve(size_t entries) override {
      map.reserve(entries);
   }

   void insertOrAssign(const K& key, const V& value) override {
      map[key] = value;
   }

   bool tryGet(const K& key, V& value) const override {
      auto it = map.find(key);
      if (it == map.end())
         return false;
      value = it->second;
      return true;
   }

   bool erase(const K& key, V& value) override {
      auto it = map.find(key);
      if (it == map.end())
         return false;
      value = it->second;
      map.erase(it);
      return true;
   }
};

template <typename K, typename V>
struct OAEntry {
   enum class State : uint8_t { Empty, Occupied, Deleted };
   K key{};
   V value{};
   State state = State::Empty;
};

template <typename K, typename V, size_t ShardCount = 64, typename HashFunc = Hasher<K>>
class ShardedOpenAddressMap : public PartitionedMap<K, V> {
   static_assert((ShardCount & (ShardCount - 1)) == 0, "ShardCount must be a power of two");

   struct Shard {
      mutable std::shared_mutex mutex;
      absl::flat_hash_map<K, V, HashFunc, std::equal_to<K>, HugePageAllocator<std::pair<const K, V>>> map;
      
      void reserve(size_t capacity) {
         map.reserve(capacity);
      }

      void insertOrAssign(const K& key, const V& value) {
         map.insert_or_assign(key, value);
      }

      bool tryGet(const K& key, V& value) const {
         auto it = map.find(key);
         if (it == map.end())
            return false;
         value = it->second;
         return true;
      }

      bool erase(const K& key, V& value) {
         auto it = map.find(key);
         if (it == map.end())
            return false;
         value = it->second;
         map.erase(it);
         return true;
      }
   };

   std::array<Shard, ShardCount> shards;
   HashFunc hasher;

   static constexpr size_t shardMask = ShardCount - 1;

   size_t shardFor(const K& key) const {
      return hasher(key) & shardMask;
   }

public:
   void reserve(size_t entries) override {
      size_t perShard = (entries + ShardCount - 1) / ShardCount;
      for (auto& shard : shards) {
         std::unique_lock lock(shard.mutex);
         shard.reserve(perShard);
      }
   }

   void insertOrAssign(const K& key, const V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      shard.insertOrAssign(key, value);
   }

   bool tryGet(const K& key, V& value) const override {
      size_t hash = hasher(key);
      auto& shard = shards[hash & shardMask];
      std::shared_lock lock(shard.mutex);
      return shard.tryGet(key, value);
   }

   bool erase(const K& key, V& value) override {
      auto& shard = shards[shardFor(key)];
      std::unique_lock lock(shard.mutex);
      return shard.erase(key, value);
   }
};

// Single-threaded specialization with no mutex overhead
template <typename K, typename V, typename HashFunc>
class ShardedOpenAddressMap<K, V, 1, HashFunc> : public PartitionedMap<K, V> {
   absl::flat_hash_map<K, V, HashFunc, std::equal_to<K>, HugePageAllocator<std::pair<const K, V>>> map;

public:
   void reserve(size_t entries) override {
      map.reserve(entries);
   }

   void insertOrAssign(const K& key, const V& value) override {
      map.insert_or_assign(key, value);
   }

   bool tryGet(const K& key, V& value) const override {
      auto it = map.find(key);
      if (it == map.end())
         return false;
      value = it->second;
      return true;
   }

   bool erase(const K& key, V& value) override {
      auto it = map.find(key);
      if (it == map.end())
         return false;
      value = it->second;
      map.erase(it);
      return true;
   }
};

// Epoch-Based Reclamation (EBR) for safe memory management in lock-free data structures
// Implementation follows Xenium's approach: https://github.com/mpoeter/xenium
class EpochManager {
public:
    struct alignas(64) ThreadState {
        std::atomic<bool> is_in_critical_region{false};
        std::atomic<uint64_t> local_epoch{0};
        std::vector<void*> retired[3];  // Circular buffer indexed by epoch % 3
        // No padding needed - alignment is sufficient
    };
    
private:
    std::atomic<uint64_t> global_epoch_{1};
    std::vector<ThreadState*> thread_states_;
    std::mutex registration_mutex_;
    
public:
    class Guard {
        ThreadState* state_;
        uint64_t epoch_;
    public:
        Guard(ThreadState* state, uint64_t epoch) : state_(state), epoch_(epoch) {
            // Mark thread as in critical region
            state_->is_in_critical_region.store(true, std::memory_order_relaxed);
            
            // (1) - seq_cst fence enforces total order and synchronizes-with acquire fence (3)
            std::atomic_thread_fence(std::memory_order_seq_cst);
            
            // Store the epoch (relaxed is safe due to fence)
            state_->local_epoch.store(epoch, std::memory_order_relaxed);
        }
        
        ~Guard() {
            // (2) - release-store synchronizes-with acquire fence (3)
            state_->is_in_critical_region.store(false, std::memory_order_seq_cst);
        }
        
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
    };
    
    ~EpochManager() {
        for (auto* state : thread_states_) {
            for (auto& retired_list : state->retired) {
                for (auto* ptr : retired_list) {
                    ::operator delete(ptr);
                }
            }
            delete state;
        }
    }
    
    ThreadState* registerThread() {
        std::lock_guard<std::mutex> lock(registration_mutex_);
        auto* state = new ThreadState();
        thread_states_.push_back(state);
        return state;
    }
    
    Guard enterCriticalSection(ThreadState* state) {
        // Load global epoch with acquire to synchronize-with release CAS (4)
        uint64_t epoch = global_epoch_.load(std::memory_order_acquire);
        return Guard(state, epoch);
    }
    
    void retire(ThreadState* state, void* ptr) {
        uint64_t current_epoch = global_epoch_.load(std::memory_order_relaxed);
        state->retired[current_epoch % 3].push_back(ptr);
        tryAdvanceEpoch(state);
    }
    
private:
    void tryAdvanceEpoch(ThreadState* state) {
        uint64_t current_epoch = global_epoch_.load(std::memory_order_relaxed);
        
        // Check if all threads have moved past the epoch from 2 epochs ago
        bool can_advance = true;
        uint64_t min_active_epoch = current_epoch;
        
        {
            std::lock_guard<std::mutex> lock(registration_mutex_);
            
            // (3) - acquire fence synchronizes-with seq_cst fence (1) and release-store (2)
            std::atomic_thread_fence(std::memory_order_acquire);
            
            for (auto* ts : thread_states_) {
                // Load operations can be relaxed due to acquire fence above
                bool in_critical = ts->is_in_critical_region.load(std::memory_order_relaxed);
                uint64_t local = ts->local_epoch.load(std::memory_order_relaxed);
                
                if (in_critical && local != current_epoch) {
                    can_advance = false;
                    break;
                }
                
                if (in_critical) {
                    min_active_epoch = std::min(min_active_epoch, local);
                }
            }
        }
        
        if (can_advance && min_active_epoch == current_epoch) {
            // (4) - release CAS synchronizes-with acquire load in enterCriticalSection
            if (global_epoch_.compare_exchange_strong(current_epoch, current_epoch + 1, 
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {
                // Safe to free memory from 2 epochs ago
                size_t old_epoch_idx = current_epoch % 3;
                for (auto* ptr : state->retired[old_epoch_idx]) {
                    ::operator delete(ptr);
                }
                state->retired[old_epoch_idx].clear();
            }
        }
    }
};

// A lock-free hash map with automatic resizing and epoch-based memory reclamation
template <typename K, typename V, typename HashFunc = Hasher<K>>
class LockFreeHashMap : public PartitionedMap<K, V> {
private:
    struct Slot {
        std::atomic<uint8_t> tag;
        K key;
        V value;
    };
    
    struct Table {
        Slot* slots;
        size_t capacity;
        size_t capacity_mask;
        std::atomic<size_t> size{0};
        
        Table(size_t cap) : capacity(cap), capacity_mask(cap - 1) {
            slots = static_cast<Slot*>(::operator new(cap * sizeof(Slot)));
            for (size_t i = 0; i < cap; ++i) {
                new (&slots[i]) Slot();
                slots[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
            }
            madvise(slots, cap * sizeof(Slot), MADV_HUGEPAGE);
        }
        
        ~Table() {
            if (slots) {
                for (size_t i = 0; i < capacity; ++i) {
                    slots[i].~Slot();
                }
                ::operator delete(slots);
            }
        }
    };

public:
    static constexpr uint8_t EMPTY_TAG = 0xFF;
    static constexpr uint8_t LOCKED_TAG = 0xFE;
    static constexpr uint8_t DELETED_TAG = 0xFD;

private:
    std::atomic<Table*> current_table_;
    std::atomic<bool> resizing_{false};  // Single writer migration lock
    
    EpochManager epoch_manager_;
    HashFunc hasher_;
    
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;
    
    // Thread-local state for automatic registration
    struct ThreadLocalState {
        EpochManager::ThreadState* ebr_state = nullptr;
        LockFreeHashMap* map = nullptr;
        
        ~ThreadLocalState() {
            // Cleanup happens in EpochManager destructor
        }
    };
    
    EpochManager::ThreadState* getThreadState() {
        // Use function-local static thread_local to avoid guard name collision
        static thread_local ThreadLocalState tls_{};
        if (tls_.ebr_state == nullptr || tls_.map != this) {
            tls_.ebr_state = epoch_manager_.registerThread();
            tls_.map = this;
        }
        return tls_.ebr_state;
    }

    static size_t nextPow2(size_t n) {
        if (n == 0) return 1;
        #if defined(__GNUC__) || defined(__clang__)
            return 1UL << (64 - __builtin_clzll(n - 1));
        #else
            size_t p = 1;
            while (p < n) p <<= 1;
            return p;
        #endif
    }
    
    // Single-writer resize: one thread does entire migration atomically
    // Readers never participate - they just read from current table
    void doResize(Table* old_table, bool force_resize = false) {
        // Only one thread can resize at a time
        bool expected = false;
        if (!resizing_.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
            // Another thread is resizing, wait for it
            while (resizing_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            return;
        }
        
        // Check if resize is actually needed (unless forced)
        if (!force_resize) {
            size_t current_size = old_table->size.load(std::memory_order_relaxed);
            if (current_size < old_table->capacity * LOAD_FACTOR_THRESHOLD) {
                resizing_.store(false, std::memory_order_release);
                return;  // No longer need resize
            }
        }
        
        // Create new table with 2x capacity
        size_t new_capacity = nextPow2(old_table->capacity * 2);
        Table* new_table = new Table(new_capacity);
        
        // Migrate all entries (single-threaded, no coordination needed)
        for (size_t i = 0; i < old_table->capacity; ++i) {
            uint8_t tag = old_table->slots[i].tag.load(std::memory_order_relaxed);
            if (tag < DELETED_TAG) {  // Valid occupied slot
                // Direct insertion into new table (no locking needed - we're the only writer)
                K key = old_table->slots[i].key;
                V value = old_table->slots[i].value;
                
                uint64_t hash = hasher_(key);
                uint8_t new_tag = hash >> 56;
                if (new_tag >= DELETED_TAG) new_tag = 0;
                
                size_t pos = hash & new_table->capacity_mask;
                while (true) {
                    if (new_table->slots[pos].tag.load(std::memory_order_relaxed) == EMPTY_TAG) {
                        new_table->slots[pos].key = key;
                        new_table->slots[pos].value = value;
                        new_table->slots[pos].tag.store(new_tag, std::memory_order_relaxed);
                        new_table->size.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                    pos = (pos + 1) & new_table->capacity_mask;
                }
            }
        }
        
        // Atomically swap to new table with release semantics
        // Readers will see the new table on their next acquire load
        auto* state = getThreadState();
        current_table_.store(new_table, std::memory_order_release);
        resizing_.store(false, std::memory_order_release);
        
        // Retire old table for safe reclamation via EBR
        epoch_manager_.retire(state, old_table);
    }

public:
    explicit LockFreeHashMap(size_t initial_capacity = 1024) {
        size_t capacity = nextPow2(initial_capacity);
        current_table_.store(new Table(capacity), std::memory_order_relaxed);
    }

    ~LockFreeHashMap() {
        Table* table = current_table_.load(std::memory_order_acquire);
        delete table;
    }

    LockFreeHashMap(const LockFreeHashMap&) = delete;
    LockFreeHashMap& operator=(const LockFreeHashMap&) = delete;

    // Pre-allocate capacity (single-writer resize)
    void reserve(size_t entries) override {
        Table* table = current_table_.load(std::memory_order_acquire);
        size_t target_capacity = nextPow2(entries);
        
        while (table->capacity < target_capacity) {
            doResize(table, true);  // Force resize regardless of load factor
            table = current_table_.load(std::memory_order_acquire);
        }
    }

    void insertOrAssign(const K& key, const V& value) override {
        auto* state = getThreadState();
        auto guard = epoch_manager_.enterCriticalSection(state);
        
        Table* table = current_table_.load(std::memory_order_acquire);
        
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        bool is_new_insert = false;

        while (true) {
            size_t pos = hash & table->capacity_mask;
            const size_t initial_pos = pos;
            
            while (true) {
                uint8_t current_tag = table->slots[pos].tag.load(std::memory_order_acquire);

                if (current_tag == tag && table->slots[pos].key == key) {
                    if (table->slots[pos].tag.compare_exchange_strong(
                            current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                        table->slots[pos].value = value;
                        table->slots[pos].tag.store(tag, std::memory_order_release);
                        return;
                    }
                    continue;
                }

                if (current_tag == EMPTY_TAG || current_tag == DELETED_TAG) {
                    if (table->slots[pos].tag.compare_exchange_strong(
                            current_tag, LOCKED_TAG, std::memory_order_acq_rel)) {
                        table->slots[pos].key = key;
                        table->slots[pos].value = value;
                        table->slots[pos].tag.store(tag, std::memory_order_release);
                        
                        if (current_tag == EMPTY_TAG) {
                            is_new_insert = true;
                        }
                        goto insertion_complete;
                    }
                    continue;
                }

                pos = (pos + 1) & table->capacity_mask;
                if (pos == initial_pos) {
                    // Table full - trigger resize
                    doResize(table);
                    // Reload table and retry
                    table = current_table_.load(std::memory_order_acquire);
                    break;  // Break inner loop to restart with new table
                }
            }
        }
        
insertion_complete:
        
        // Update size and check if resize needed
        if (is_new_insert) {
            size_t new_size = table->size.fetch_add(1, std::memory_order_relaxed) + 1;
            if (new_size > table->capacity * LOAD_FACTOR_THRESHOLD) {
                doResize(table);
            }
        }
    }

    bool tryGet(const K& key, V& value) const override {
        // Note: const_cast is safe here as we're only modifying thread_local state
        auto* state = const_cast<LockFreeHashMap*>(this)->getThreadState();
        auto guard = const_cast<LockFreeHashMap*>(this)->epoch_manager_.enterCriticalSection(state);
        
        Table* table = current_table_.load(std::memory_order_acquire);

        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & table->capacity_mask;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = table->slots[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == tag && table->slots[pos].key == key) {
                value = table->slots[pos].value;
                // Double-check tag to ensure slot wasn't modified during read
                if (table->slots[pos].tag.load(std::memory_order_acquire) == tag) {
                    return true;
                }
            }
            
            if (current_tag == EMPTY_TAG) {
                return false;
            }

            pos = (pos + 1) & table->capacity_mask;

        } while (pos != initial_pos);
        
        return false;
    }

    bool erase(const K& key, V& value) override {
        auto* state = getThreadState();
        auto guard = epoch_manager_.enterCriticalSection(state);
        
        Table* table = current_table_.load(std::memory_order_acquire);
        
        uint64_t hash = hasher_(key);
        uint8_t tag = hash >> 56;
        if (tag >= DELETED_TAG) tag = 0;

        size_t pos = hash & table->capacity_mask;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = table->slots[pos].tag.load(std::memory_order_acquire);
            
            if (current_tag == tag && table->slots[pos].key == key) {
                if (table->slots[pos].tag.compare_exchange_strong(
                        current_tag, DELETED_TAG, std::memory_order_acq_rel)) {
                    value = table->slots[pos].value;
                    table->size.fetch_sub(1, std::memory_order_relaxed);
                    return true;
                }
                continue;
            }
            
            if (current_tag == EMPTY_TAG) {
                return false;
            }

            pos = (pos + 1) & table->capacity_mask;
        } while (pos != initial_pos);

        return false;
    }
};
