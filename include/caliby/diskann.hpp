#ifndef DISKANN_HPP
#define DISKANN_HPP

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "aligned_allocator.hpp"  // Include our new allocator
#include "calico.hpp"
#include "utils.hpp"

// Include the official DiskANN distance header
#include "distance_diskann.hpp"

// Forward declaration of the ThreadPool class
class ThreadPool;

// --- Neighbor struct, essential for NeighborPriorityQueue ---
struct Neighbor {
    uint32_t id;
    float distance;
    bool expanded;  // Flag to indicate if the node has been expanded.

    Neighbor() = default;
    Neighbor(uint32_t id, float distance) : id(id), distance(distance), expanded(false) {}

    inline bool operator<(const Neighbor& other) const { return distance < other.distance; }
    inline bool operator>(const Neighbor& other) const { return distance > other.distance; }
};

// --- Abstract Base Class for Python Binding (Type Erasure) ---
class DiskANNBase {
   public:
    virtual ~DiskANNBase() = default;

    struct BuildParams {
        size_t L_build = 100;
        float alpha = 1.2f;
        size_t num_threads = 0;
    };
    struct SearchParams {
        size_t L_search;
        size_t beam_width = 4;  // Reused for prefetching width
    };

    virtual void build(const void* data, const std::vector<std::vector<uint32_t>>& tags, uint64_t num_points,
                       const BuildParams& params) = 0;
    virtual void optimize_layout() = 0;
    virtual std::vector<std::pair<float, uint32_t>> search(const void* query, size_t K, const SearchParams& params) = 0;
    virtual std::vector<std::pair<float, uint32_t>> search_with_filter(const void* query, uint32_t filter_label,
                                                                       size_t K, const SearchParams& params) = 0;
    virtual void insert_point(const void* point, const std::vector<uint32_t>& tags, uint32_t external_id) = 0;
    virtual void lazy_delete(uint32_t external_id) = 0;
    virtual void consolidate_deletes(const BuildParams& params) = 0;
    virtual size_t get_dimensions() const = 0;
    virtual size_t get_R() const = 0;
    virtual std::vector<std::vector<std::pair<float, uint32_t>>> search_parallel(const void* queries,
                                                                                 size_t num_queries, size_t K,
                                                                                 const SearchParams& params,
                                                                                 size_t num_threads) = 0;
    virtual std::vector<std::vector<std::pair<float, uint32_t>>> search_with_filter_parallel(
        const void* queries, size_t num_queries, uint32_t filter_label, size_t K, const SearchParams& params,
        size_t num_threads) = 0;
};

std::unique_ptr<DiskANNBase> create_index(size_t dimensions, uint64_t max_elements, size_t R_max_degree,
                                          bool is_dynamic, uint32_t index_id = 0);

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode = 8>
class DiskANN : public DiskANNBase {
    static_assert(std::is_same_v<TagT, uint32_t>, "TagT must be uint32_t to implement the DiskANNBase interface.");
    static_assert(std::is_same_v<T, float>, "This implementation is optimized for float type.");

   public:
    DiskANN(uint64_t max_elements, size_t R_max_degree, bool is_dynamic, uint32_t index_id = 0);
    ~DiskANN();

    void build(const void* data, const std::vector<std::vector<uint32_t>>& tags, uint64_t num_points,
               const BuildParams& params) override;
    void optimize_layout() override;
    std::vector<std::pair<float, uint32_t>> search(const void* query, size_t K, const SearchParams& params) override;
    std::vector<std::pair<float, uint32_t>> search_with_filter(const void* query, uint32_t filter_label, size_t K,
                                                               const SearchParams& params) override;
    void insert_point(const void* point, const std::vector<uint32_t>& tags, uint32_t external_id) override;
    void lazy_delete(uint32_t external_id) override;
    void consolidate_deletes(const BuildParams& params) override;
    size_t get_dimensions() const override { return Dim; }
    size_t get_R() const override { return _R_max_degree; }
    std::vector<std::vector<std::pair<float, uint32_t>>> search_parallel(const void* queries, size_t num_queries,
                                                                         size_t K, const SearchParams& params,
                                                                         size_t num_threads) override;
    std::vector<std::vector<std::pair<float, uint32_t>>> search_with_filter_parallel(const void* queries,
                                                                                     size_t num_queries,
                                                                                     uint32_t filter_label, size_t K,
                                                                                     const SearchParams& params,
                                                                                     size_t num_threads) override;

    void build_typed(const T* data, const std::vector<std::vector<TagT>>& tags, uint64_t num_points,
                     const BuildParams& params);
    std::vector<std::pair<float, uint32_t>> search_typed(const T* query, size_t K, const SearchParams& params);
    std::vector<std::pair<float, uint32_t>> search_with_filter_typed(const T* query, const TagT& filter_label, size_t K,
                                                                     const SearchParams& params);
    void insert_point_typed(const T* point, const std::vector<TagT>& tags, uint32_t external_id);
    std::vector<std::vector<std::pair<float, uint32_t>>> search_parallel_typed(const T* queries, size_t num_queries,
                                                                               size_t K, const SearchParams& params,
                                                                               size_t num_threads);
    std::vector<std::vector<std::pair<float, uint32_t>>> search_with_filter_parallel_typed(
        const T* queries, size_t num_queries, const TagT& filter_label, size_t K, const SearchParams& params,
        size_t num_threads);

   private:
    // --- Official DiskANN NeighborPriorityQueue implementation ---
    class NeighborPriorityQueue {
       public:
        NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0) {}
        explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1) {}

        void insert(const Neighbor& nbr) {
            if (_size == _capacity && _data[_size - 1] < nbr) {
                return;
            }
            size_t lo = 0, hi = _size;
            while (lo < hi) {
                size_t mid = (lo + hi) >> 1;
                if (nbr < _data[mid]) {
                    hi = mid;
                } else if (_data[mid].id == nbr.id) {
                    return;
                } else {
                    lo = mid + 1;
                }
            }
            if (lo < _capacity) {
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
            }
            _data[lo] = nbr;
            if (_size < _capacity) {
                _size++;
            }
            if (lo < _cur) {
                _cur = lo;
            }
        }
        Neighbor closest_unexpanded() {
            _data[_cur].expanded = true;
            size_t pre = _cur;
            while (_cur < _size && _data[_cur].expanded) {
                _cur++;
            }
            return _data[pre];
        }
        bool has_unexpanded_node() const { return _cur < _size; }
        size_t size() const { return _size; }
        void reserve(size_t capacity) {
            if (capacity + 1 > _data.size()) {
                _data.resize(capacity + 1);
            }
            _capacity = capacity;
        }
        Neighbor& operator[](size_t i) { return _data[i]; }
        const Neighbor& operator[](size_t i) const { return _data[i]; }
        void clear() {
            _size = 0;
            _cur = 0;
        }

        friend class DiskANN;

       private:
        size_t _size, _capacity, _cur;
        std::vector<Neighbor> _data;
    };

    struct DiskANNMetadataPage {
        bool dirty;
        PID base_pid;
        uint64_t max_elements;
        std::atomic<uint64_t> node_count;
        std::atomic<uint64_t> alloc_count{0};  // Per-index allocation counter
        bool is_layout_optimized;
    };

    struct VamanaPage {
        bool dirty;
        uint16_t node_count_in_page;
        uint16_t padding[3];
        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count_in_page) + sizeof(padding);
        uint8_t* getNodeData() { return reinterpret_cast<uint8_t*>(this) + HeaderSize; }
        const uint8_t* getNodeData() const { return reinterpret_cast<const uint8_t*>(this) + HeaderSize; }
    };

    static constexpr size_t VectorSize = Dim * sizeof(T);
    static constexpr size_t TagsHeaderSize = sizeof(uint16_t);
    static constexpr size_t TagsDataSize = MaxTagsPerNode * sizeof(TagT);
    static constexpr size_t NeighborsHeaderSize = sizeof(uint16_t);

    size_t _NeighborsDataSize;
    size_t _FixedNodeSize;
    size_t _NodesPerPage;

    class NodeAccessor {
       protected:
        const uint8_t* _node_start;
        const DiskANN* _index;

       public:
        NodeAccessor(const VamanaPage* page, uint32_t node_idx, const DiskANN* index);
        const T* getVector() const;
        std::span<const TagT> getTags() const;
        std::span<const uint32_t> getNeighbors() const;
    };

    class MutableNodeAccessor : public NodeAccessor {
       private:
        VamanaPage* _page;

       public:
        MutableNodeAccessor(VamanaPage* page, uint32_t node_idx, const DiskANN* index);
        void setVector(const T* vector);
        void setTags(const std::vector<TagT>& tags);
        void setNeighbors(const std::vector<uint32_t>& neighbors);
    };

    const bool _is_dynamic;
    uint32_t _index_id;  // Index ID for multi-index support
    PIDAllocator* _allocator;  // Per-index allocator for page allocation
    PID _metadata_pid;
    PID _base_pid;
    
    // Helper to encode local PID with index_id
    inline PID makeGlobalPID(PID local_pid) const {
        return (static_cast<PID>(_index_id) << 32) | (local_pid & 0xFFFFFFFFULL);
    }
    uint64_t _max_elements;
    size_t _R_max_degree;
    bool _is_layout_optimized;

    std::unordered_map<TagT, uint32_t> _medoids;
    std::unordered_map<uint32_t, uint32_t> _external_to_internal_map;
    std::vector<uint32_t> _internal_to_external_map;
    std::shared_mutex _map_lock;
    ConcurrentSet<uint32_t> _deleted_nodes;
    std::unique_ptr<VisitedListPool> _visited_list_pool;

    // NEW: Cache for medoid vectors to avoid lock contention at search start
    std::unordered_map<TagT, std::vector<T, AlignedAllocator<T, 32>>> _medoid_vectors_cache;

    ThreadPool* getOrCreateThreadPool(size_t num_threads) const;
    mutable std::unique_ptr<ThreadPool> _thread_pool;
    mutable std::mutex _pool_mutex;

    // Thread pools for parallel operations - reused across calls
    mutable std::unique_ptr<class ThreadPool> search_thread_pool_{nullptr};
    mutable size_t search_pool_size_{0};
    mutable std::mutex search_pool_mutex_;

    mutable std::unique_ptr<class ThreadPool> add_thread_pool_{nullptr};
    mutable size_t add_pool_size_{0};
    mutable std::mutex add_pool_mutex_;

   private:
    inline PID getNodePID(uint32_t node_id) const;
    inline uint32_t getNodeIndexInPage(uint32_t node_id) const;

    bool getNodeVector(uint32_t node_id, T* dest_vector) const;
    bool getNodeTags(uint32_t node_id, std::vector<TagT>& dest_tags) const;
    bool getNodeNeighbors(uint32_t node_id, std::vector<uint32_t>& dest_neighbors) const;

    struct NodeDataBatch {
        std::vector<T, AlignedAllocator<T, 32>> vectors;
        std::vector<TagT> tags;
        std::vector<uint16_t> tag_counts;
        void clear() {
            vectors.clear();
            tags.clear();
            tag_counts.clear();
        }
    };

    bool get_nodes_data_batch(const std::vector<uint32_t>& ids, NodeDataBatch& data_batch) const;
    bool get_vectors_batch(const std::vector<uint32_t>& ids, T* dest_buffer) const;
    
    // Compute distances directly from page memory without copying vectors
    bool compute_distances_batch(const std::vector<uint32_t>& ids, const T* query,
                                 std::vector<std::pair<float, uint32_t>>& distances_out) const;

    // MODIFIED: Signature updated to accept pre-fetched start node vector
    std::vector<std::pair<float, uint32_t>> greedy_search_final(uint32_t start_node_id, const T* start_node_vec,
                                                                const T* query, size_t K, const SearchParams& params,
                                                                const TagT* filter_label);

    std::vector<std::pair<float, uint32_t>> greedy_search_original(const std::vector<uint32_t>& start_nodes,
                                                                   const T* query, size_t K, const SearchParams& params,
                                                                   const TagT* filter_label);

    void robust_prune_batched(uint32_t p_id, std::vector<std::pair<float, uint32_t>>& candidates, float alpha,
                              bool filter_aware);
    void connect_neighbors_batched(uint32_t node_id, const std::vector<uint32_t>& neighbors, const BuildParams& params,
                                   bool filter_aware);
    void compute_medoids(uint64_t num_points);

    // Helper method to get or create thread pool
    class ThreadPool* getOrCreateSearchPool(size_t num_threads) const;
    class ThreadPool* getOrCreateAddPool(size_t num_threads) const;
};

#endif  // DISKANN_HPP