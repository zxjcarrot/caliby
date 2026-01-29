#ifndef HNSW_HPP
#define HNSW_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>  // Required for std::numeric_limits
#include <optional>
#include <queue>
#include <random>
#include <span>
#include <vector>
#include <iomanip>  // For std::setprecision
#include <mutex>

#include "calico.hpp"
#include "distance.hpp"

// Forward declaration
class ThreadPool;

// --- HNSW Metadata ---

#include "utils.hpp"

// --- HNSW Stats ---
struct HNSWStats {
    // Live counters (updated during operations)
    std::atomic<u64> dist_comps{0};
    std::atomic<u64> search_hops{0};  // Total nodes visited/evaluated during search

    // Snapshot stats (calculated by iterating the graph)
    u32 num_levels = 0;
    std::vector<u64> nodes_per_level;
    std::vector<u64> links_per_level;
    double avg_neighbors_total = 0.0;
    std::vector<double> avg_neighbors_per_level;

    // Default constructor
    HNSWStats() = default;

    // Custom Copy Constructor
    HNSWStats(const HNSWStats& other) {
        dist_comps.store(other.dist_comps.load());
        search_hops.store(other.search_hops.load());
        num_levels = other.num_levels;
        nodes_per_level = other.nodes_per_level;
        links_per_level = other.links_per_level;
        avg_neighbors_total = other.avg_neighbors_total;
        avg_neighbors_per_level = other.avg_neighbors_per_level;
    }

    // Custom Copy Assignment Operator
    HNSWStats& operator=(const HNSWStats& other) {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            search_hops.store(other.search_hops.load());
            num_levels = other.num_levels;
            nodes_per_level = other.nodes_per_level;
            links_per_level = other.links_per_level;
            avg_neighbors_total = other.avg_neighbors_total;
            avg_neighbors_per_level = other.avg_neighbors_per_level;
        }
        return *this;
    }

    // We can also define move operations for efficiency, although the copy constructor will solve the immediate
    // problem. Custom Move Constructor
    HNSWStats(HNSWStats&& other) noexcept {
        dist_comps.store(other.dist_comps.load());
        search_hops.store(other.search_hops.load());
        num_levels = other.num_levels;
        nodes_per_level = std::move(other.nodes_per_level);
        links_per_level = std::move(other.links_per_level);
        avg_neighbors_total = other.avg_neighbors_total;
        avg_neighbors_per_level = std::move(other.avg_neighbors_per_level);
    }

    // Custom Move Assignment Operator
    HNSWStats& operator=(HNSWStats&& other) noexcept {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            search_hops.store(other.search_hops.load());
            num_levels = other.num_levels;
            nodes_per_level = std::move(other.nodes_per_level);
            links_per_level = std::move(other.links_per_level);
            avg_neighbors_total = other.avg_neighbors_total;
            avg_neighbors_per_level = std::move(other.avg_neighbors_per_level);
        }
        return *this;
    }

    // Reset live counters
    void reset_live_counters() {
        dist_comps.store(0);
        search_hops.store(0);
    }

    // Format stats into a readable string
    std::string toString() const {
        std::ostringstream oss;
        oss << "--- HNSW Performance Stats ---\n";
        oss << "Distance Computations: " << dist_comps.load() << "\n";
        oss << "Search Hops:           " << search_hops.load() << "\n";
        oss << "\n--- Graph Structure Stats ---\n";
        oss << "Number of Levels: " << num_levels << "\n";

        u64 total_nodes = 0;
        u64 total_links = 0;

        for (u32 i = 0; i < num_levels; ++i) {
            oss << "Level " << i << ":\n";
            oss << "  Nodes: " << nodes_per_level[i] << "\n";
            oss << "  Avg Neighbors: " << std::fixed << std::setprecision(2) << avg_neighbors_per_level[i] << "\n";
            total_nodes += nodes_per_level[i];
            total_links += links_per_level[i];
        }

        oss << "\n--- Overall Graph Stats ---\n";
        oss << "Total Nodes: " << total_nodes << "\n";
        oss << "Total Links: " << total_links << "\n";
        oss << "Avg Neighbor List Length (Total): " << std::fixed << std::setprecision(2) << avg_neighbors_total
            << "\n";

        return oss.str();
    }
};

// --- HNSW Metadata ---
struct HNSWMetadataPage {
    bool dirty;
    PID base_pid;                 // The starting Page ID for HNSW data pages.
    u64 max_elements;             // The maximum number of elements the index can hold.
    std::atomic<u64> node_count;  // The total number of nodes currently inserted.
    std::atomic<u64> alloc_count{0};  // Per-index allocation counter

    // Entry point information.
    u32 enter_point_node_id;     // The global entry point node ID for the graph.
    std::atomic<u32> max_level;  // The highest level currently present in the graph.
    static constexpr u32 invalid_node_id = std::numeric_limits<u32>::max();
};

// --- HNSW Main Class ---

template <typename DistanceMetric = hnsw_distance::SIMDAcceleratedL2>
class HNSW {
   public:
    // Fixed-size node layout (no more slots/indirection):
    // Each node has: [Vector Data][Level Counts][Neighbor IDs]
    // Node size is constant and computed at runtime

    // --- Runtime parameters (formerly template parameters) ---
    const size_t Dim;             // Vector dimension (runtime constant)
    const size_t M;               // Max number of neighbors for layers > 0
    const size_t M0;              // Max number of neighbors for layer 0 (computed as 2*M)
    const size_t efConstruction;  // Number of neighbors to search during index construction
    const size_t MaxLevel;        // Pre-estimated max level for random level generation

    // --- Runtime size calculations for fixed-size nodes ---
    const size_t MaxNeighborsHeaderSize;  // MaxLevel * sizeof(u16) - level counts
    const size_t MaxNeighborsListSize;    // (M0 + (MaxLevel - 1) * M) * sizeof(u32) - neighbor IDs
    const size_t VectorSize;                // Dim * sizeof(float)
    const size_t FixedNodeSize;  // VectorSize + MaxNeighborsHeaderSize + MaxNeighborsListSize

    // HNSW data page: uses fixed-size node grid layout for direct access.
    struct HNSWPage {
        bool dirty;
        u16 node_count;
        u16 padding[3];  // Align to 8 bytes

        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count) + sizeof(padding);

        // Get node data area (nodes start immediately after header)
        u8* getNodeData() { return reinterpret_cast<u8*>(this) + HeaderSize; }

        const u8* getNodeData() const { return reinterpret_cast<const u8*>(this) + HeaderSize; }
    };

    const size_t MaxNodesPerPage;  // Runtime computed: (pageSize - HNSWPage::HeaderSize) / FixedNodeSize
    const size_t NodesPerPage;     // Same as MaxNodesPerPage

    // Calculates and returns a snapshot of the graph structure stats,
    // along with the current values of live counters.
    HNSWStats getStats() const;

    // Resets live counters like distance computations and search hops.
    void resetStats() { stats_.reset_live_counters(); }

   private:
    uint32_t index_id_;  // Index ID for multi-index support
    std::string name_;   // Optional name for the index
    PIDAllocator* allocator_;  // Per-index allocator for page allocation
    PID metadata_pid;
    PID base_pid;
    
    // Helper to encode local PID with index_id
    inline PID makeGlobalPID(PID local_pid) const {
        return (static_cast<PID>(index_id_) << 32) | (local_pid & 0xFFFFFFFFULL);
    }
    u64 max_elements_;
    double mult_factor;
    bool enable_prefetch_;
    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};
    bool recovered_from_disk_ = false;
    
    // `mutable` allows const methods like searchKnn to update stats.
    mutable HNSWStats stats_;
    
    // Thread pools for parallel operations - reused across calls
    mutable std::unique_ptr<class ThreadPool> search_thread_pool_{nullptr};
    mutable size_t search_pool_size_{0};
    mutable std::mutex search_pool_mutex_;
    
    mutable std::unique_ptr<class ThreadPool> add_thread_pool_{nullptr};
    mutable size_t add_pool_size_{0};
    mutable std::mutex add_pool_mutex_;
    
    // Helper method to get or create thread pool
    class ThreadPool* getOrCreateSearchPool(size_t num_threads) const;
    class ThreadPool* getOrCreateAddPool(size_t num_threads) const;

   public:
    // Constructor with runtime parameters
    HNSW(u64 max_elements, size_t dim, size_t M = 16, size_t ef_construction = 200, bool enable_prefetch = true,
        bool skip_recovery = false, uint32_t index_id = 0, const std::string& name = "");
    // Destructor - must be defined in .cpp where ThreadPool is complete
    ~HNSW();
    
    // Get the name of the index
    const std::string& getName() const { return name_; }

   private:
    // Helper function to estimate MaxLevel from max_elements
    static size_t estimateMaxLevel(u64 max_elements, size_t M);

    inline float calculateDistance(const float* v1, const float* v2) const {
        return DistanceMetric::compare(v1, v2, Dim);
    }

   public:
    // Add a point to the index (auto-assigns sequential node_id).
    void addPoint(const float* point, u32& node_id_out);
    
    // Add a point with a specific node_id (e.g., doc_id). 
    // The caller is responsible for ensuring node_id is unique and within max_elements.
    void addPointWithId(const float* point, u32 node_id);

    // Search for the K nearest neighbors for a single query.
    template <bool stats = false>
    std::vector<std::pair<float, u32>> searchKnn(const float* query, size_t k, size_t ef_search_param);

    // Search for the K nearest neighbors for a batch of queries in parallel.
    template <bool stats = false>
    std::vector<std::vector<std::pair<float, u32>>> searchKnn_parallel(std::span<const float> queries, size_t k,
                                                                       size_t ef_search_param, size_t num_threads = 0);
    // Add a batch of points to the index in parallel.
    void addPoint_parallel(std::span<const float> points, size_t num_threads = 0);
    
    // Add a batch of points with specific IDs in parallel.
    // Useful for Collection where doc_id must be used as node_id.
    void addPointsWithIdsParallel(const std::vector<const float*>& data_ptrs,
                                  const std::vector<uint32_t>& ids,
                                  size_t num_threads = 0);

    HNSW(const HNSW&) = delete;
    HNSW& operator=(const HNSW&) = delete;
    HNSW(HNSW&&) = delete;
    HNSW& operator=(HNSW&&) = delete;
    PID getMetadataPid() const { return metadata_pid; }
    size_t getDim() const { return Dim; }

    // Get buffer manager statistics
    u64 getBufferManagerAllocCount() const;

    // Print comprehensive index information
    std::string getIndexInfo() const;

    bool wasRecovered() const { return recovered_from_disk_; }

   private:
    // --- Internal Helper Classes & Functions ---
    inline PID getNodePID(u32 node_id) const;
    inline u32 getNodeIndexInPage(u32 node_id) const;

    class NodeAccessor {
       private:
        const HNSWPage* page;
        u32 node_index;
        const HNSW* hnsw;
        const size_t node_offset;  // Precomputed node_index * FixedNodeSize

       public:
        NodeAccessor(const HNSWPage* p, u32 node_idx, const HNSW* hnsw_instance)
            : page(p), node_index(node_idx), hnsw(hnsw_instance), node_offset(node_idx * hnsw_instance->FixedNodeSize) {}

        u32 getLevel() const {
            // Level is stored as the maximum non-zero count in the level counts array
            const u16* counts = getLevelCounts();
            u32 max_level = 0;
            for (u32 i = 0; i < hnsw->MaxLevel; ++i) {
                if (counts[i] > 0) max_level = i;
            }
            return max_level;
        }

        const float* getVector() const {
            const u8* node_start = page->getNodeData() + node_offset;
            return reinterpret_cast<const float*>(node_start);
        }

        static inline u32 getVectorOffset(HNSW* hnsw, u32 graph_node_id) {
            return HNSWPage::HeaderSize + hnsw->getNodeIndexInPage(graph_node_id) * hnsw->FixedNodeSize;
        }

        static inline u32 getL0NeighborOffset(HNSW* hnsw, u32 graph_node_id) {
            return HNSWPage::HeaderSize + hnsw->getNodeIndexInPage(graph_node_id) * hnsw->FixedNodeSize +
                   hnsw->VectorSize + hnsw->MaxNeighborsHeaderSize;
        }

        const u16* getLevelCounts() const {
            const u8* node_start = page->getNodeData() + node_offset;
            return reinterpret_cast<const u16*>(node_start + hnsw->VectorSize);
        }

        std::span<const u32> getNeighbors(u32 level, const HNSW* hnsw_instance) const {
            // optimize for level 0 access
            if (level == 0) {
                // directly return level 0 neighbors
                const u8* node_start = page->getNodeData() + node_offset;
                const u32* neighbor_ids_start =
                    reinterpret_cast<const u32*>(node_start + hnsw_instance->VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
                u16 count_for_level = getLevelCounts()[0];
                return {neighbor_ids_start, count_for_level};
            }
            // General case for levels > 0
            const u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return {};

            u32 level_neighbor_offset_count = (level == 0) ? 0 : (u32)(hnsw_instance->M0 + (level - 1) * hnsw_instance->M);

            const u8* node_start = page->getNodeData() + node_offset;
            const u32* neighbor_ids_start =
                reinterpret_cast<const u32*>(node_start + hnsw_instance->VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
            const u32* level_start_ptr = neighbor_ids_start + level_neighbor_offset_count;
            u16 count_for_level = counts_per_level[level];
            return {level_start_ptr, count_for_level};
        }
    };

    // Non-const version for modification
    class MutableNodeAccessor {
       private:
        HNSWPage* page;
        u32 node_index;
        const HNSW* hnsw;
        const size_t node_offset;  // Precomputed node_index * FixedNodeSize

       public:
        MutableNodeAccessor(HNSWPage* p, u32 node_idx, const HNSW* hnsw_instance)
            : page(p), node_index(node_idx), hnsw(hnsw_instance), node_offset(node_idx * hnsw_instance->FixedNodeSize) {}

        const float* getVector() const {
            const u8* node_start = page->getNodeData() + node_offset;
            return reinterpret_cast<const float*>(node_start);
        }

        float* getVector() {
            u8* node_start = page->getNodeData() + node_offset;
            return reinterpret_cast<float*>(node_start);
        }

        u16* getLevelCounts() {
            u8* node_start = page->getNodeData() + node_offset;
            return reinterpret_cast<u16*>(node_start + hnsw->VectorSize);
        }

        const u16* getLevelCounts() const {
            const u8* node_start = page->getNodeData() + node_offset;
            return reinterpret_cast<const u16*>(node_start + hnsw->VectorSize);
        }

        std::span<const u32> getNeighbors(u32 level, const HNSW* hnsw_instance) const {
            const u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return {};

            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            const u8* node_start = page->getNodeData() + node_offset;
            const u32* neighbor_ids_start =
                reinterpret_cast<const u32*>(node_start + hnsw_instance->VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
            const u32* level_start_ptr = neighbor_ids_start + level_neighbor_offset_count;
            u16 count_for_level = counts_per_level[level];
            return {level_start_ptr, count_for_level};
        }

        void setNeighbors(u32 level, const std::vector<u32>& new_neighbors, const HNSW* hnsw_instance) {
            u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return;

            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            u8* node_start = page->getNodeData() + node_offset;
            u32* neighbor_ids_start =
                reinterpret_cast<u32*>(node_start + hnsw_instance->VectorSize + hnsw_instance->MaxNeighborsHeaderSize);
            u32* level_start_ptr = neighbor_ids_start + level_neighbor_offset_count;
            size_t M_level = (level == 0) ? hnsw_instance->M0 : hnsw_instance->M;

            size_t count_to_copy = std::min(new_neighbors.size(), M_level);
            std::copy(new_neighbors.begin(), new_neighbors.begin() + count_to_copy, level_start_ptr);
            counts_per_level[level] = count_to_copy;
            page->dirty = true;
        }

        bool addNeighbor(u32 level, u32 new_neighbor_id, const HNSW* hnsw_instance) {
            u16* counts_per_level = getLevelCounts();
            if (level >= hnsw_instance->MaxLevel) return false;

            size_t M_level = (level == 0) ? hnsw_instance->M0 : hnsw_instance->M;
            u16 current_count = counts_per_level[level];

            // Check if there is space
            if (current_count >= M_level) {
                return false;
            }

            // Calculate the memory offset to the start of the neighbor list for this level
            u32 level_neighbor_offset_count = 0;
            if (level > 0) {
                level_neighbor_offset_count += hnsw_instance->M0;
                for (u32 i = 1; i < level; ++i) {
                    level_neighbor_offset_count += hnsw_instance->M;
                }
            }

            u8* node_start = page->getNodeData() + node_offset;
            u32* neighbor_ids_start =
                reinterpret_cast<u32*>(node_start + hnsw_instance->VectorSize + hnsw_instance->MaxNeighborsHeaderSize);

            // Get the pointer to the exact location to append the new neighbor
            u32* append_location = neighbor_ids_start + level_neighbor_offset_count + current_count;

            // Write the new neighbor ID directly into the page
            *append_location = new_neighbor_id;

            // Increment the count for this level
            counts_per_level[level]++;
            page->dirty = true;

            return true;
        }
    };

    u32 getRandomLevel();
    template <bool stats = false>
    std::vector<std::pair<float, u32>> searchLayer(const float* query, u32 entry_point_id, u32 level, size_t ef,
                                                   std::optional<std::pair<float, u32>> initial_entry_dist_pair);
    template <bool stats = false>
    std::pair<float, u32> findBestEntryPointForLevel(const float* query, u32 entry_point_id, int level,
                                                     float entry_point_dist);
    template <bool stats = false>
    std::pair<float, u32> searchBaseLayer(const float* query, u32 entry_point_id, int start_level, int end_level);

    // --- Now takes the query vector to match hnswlib's heuristic ---
    std::vector<std::pair<float, u32>> selectNeighborsHeuristic(const float* query,
                                                                const std::vector<std::pair<float, u32>>& candidates,
                                                                size_t M_limit);

    void addPoint_internal(const float* point, u32 node_id);
};

#endif  // HNSW_HPP