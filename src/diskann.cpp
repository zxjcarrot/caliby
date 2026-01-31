#include "diskann.hpp"
#include "logging.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__) && !defined(CALIBY_NO_AVX2)
#include <immintrin.h>
#endif

#include "calico.hpp"

static Distance distance_metric = Distance(Metric::L2);

// ========================================================================
// Factory Function, Accessors, and ThreadPool
// ========================================================================
std::unique_ptr<DiskANNBase> create_index(size_t dimensions, uint64_t max_elements, size_t R_max_degree,
                                          bool is_dynamic, uint32_t index_id) {
    switch (dimensions) {
        case 16:
            return std::make_unique<DiskANN<float, uint32_t, 16>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 32:
            return std::make_unique<DiskANN<float, uint32_t, 32>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 64:
            return std::make_unique<DiskANN<float, uint32_t, 64>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 96:
            return std::make_unique<DiskANN<float, uint32_t, 96>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 128:
            return std::make_unique<DiskANN<float, uint32_t, 128>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 256:
            return std::make_unique<DiskANN<float, uint32_t, 256>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 384:
            return std::make_unique<DiskANN<float, uint32_t, 384>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 512:
            return std::make_unique<DiskANN<float, uint32_t, 512>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 768:
            return std::make_unique<DiskANN<float, uint32_t, 768>>(max_elements, R_max_degree, is_dynamic, index_id);
        case 1024:
            return std::make_unique<DiskANN<float, uint32_t, 1024>>(max_elements, R_max_degree, is_dynamic, index_id);
        default:
            throw std::runtime_error("Unsupported dimension: " + std::to_string(dimensions) + ". Please recompile.");
    }
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::NodeAccessor(const VamanaPage* page, uint32_t node_idx,
                                                                  const DiskANN* index)
    : _node_start(page->getNodeData() + node_idx * index->_FixedNodeSize), _index(index) {}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
const T* DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::getVector() const {
    return reinterpret_cast<const T*>(_node_start);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::span<const TagT> DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::getTags() const {
    const auto* count_ptr = reinterpret_cast<const uint16_t*>(_node_start + VectorSize);
    const auto* tags_start = reinterpret_cast<const TagT*>(count_ptr + 1);
    return {tags_start, *count_ptr};
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::span<const uint32_t> DiskANN<T, TagT, Dim, MaxTagsPerNode>::NodeAccessor::getNeighbors() const {
    const auto* count_ptr = reinterpret_cast<const uint16_t*>(_node_start + VectorSize + TagsHeaderSize + TagsDataSize);
    const auto* neighbors_start = reinterpret_cast<const uint32_t*>(count_ptr + 1);
    return {neighbors_start, *count_ptr};
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::MutableNodeAccessor(VamanaPage* page, uint32_t node_idx,
                                                                                const DiskANN* index)
    : NodeAccessor(page, node_idx, index), _page(page) {}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::setVector(const T* vector) {
    memcpy(const_cast<uint8_t*>(this->_node_start), vector, VectorSize);
    _page->dirty = true;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::setTags(const std::vector<TagT>& tags) {
    auto* count_ptr = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(this->_node_start) + VectorSize);
    auto* tags_start = reinterpret_cast<TagT*>(count_ptr + 1);
    size_t count_to_copy = std::min(tags.size(), MaxTagsPerNode);
    *count_ptr = static_cast<uint16_t>(count_to_copy);
    memcpy(tags_start, tags.data(), count_to_copy * sizeof(TagT));
    _page->dirty = true;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::MutableNodeAccessor::setNeighbors(const std::vector<uint32_t>& neighbors) {
    auto* count_ptr = reinterpret_cast<uint16_t*>(const_cast<uint8_t*>(this->_node_start) + VectorSize +
                                                  TagsHeaderSize + TagsDataSize);
    auto* neighbors_start = reinterpret_cast<uint32_t*>(count_ptr + 1);
    size_t count_to_copy = std::min(neighbors.size(), this->_index->_R_max_degree);
    *count_ptr = static_cast<uint16_t>(count_to_copy);
    memcpy(neighbors_start, neighbors.data(), count_to_copy * sizeof(uint32_t));
    _page->dirty = true;
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
ThreadPool* DiskANN<T, TagT, Dim, MaxTagsPerNode>::getOrCreateThreadPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(_pool_mutex);
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    if (!_thread_pool || _thread_pool->getThreadCount() != threads_to_use) {
        _thread_pool = std::make_unique<ThreadPool>(threads_to_use);
    }
    return _thread_pool.get();
}

// ========================================================================
// Constructor & Destructor
// ========================================================================

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::DiskANN(uint64_t max_elements, size_t R_max_degree, bool is_dynamic, uint32_t index_id)
    : _is_dynamic(is_dynamic), _index_id(index_id), _max_elements(max_elements), _R_max_degree(R_max_degree), _is_layout_optimized(false) {
    _NeighborsDataSize = _R_max_degree * sizeof(uint32_t);
    size_t calculated_size = VectorSize + TagsHeaderSize + TagsDataSize + NeighborsHeaderSize + _NeighborsDataSize;
    _FixedNodeSize = (calculated_size + 63) & ~63;  // 64B align

    _NodesPerPage = (pageSize - VamanaPage::HeaderSize) / _FixedNodeSize;
    if (_NodesPerPage == 0) throw std::runtime_error("Page size too small for node configuration.");

    CALIBY_LOG_INFO("DiskANN", "Creating new index... FixedNodeSize=", _FixedNodeSize, ", NodesPerPage=", _NodesPerPage);
    
    // Get or create PIDAllocator for this index
    uint64_t num_pages = (_max_elements + _NodesPerPage - 1) / _NodesPerPage;
    u64 total_pages_needed = 1 + num_pages;  // 1 metadata + data pages
    _allocator = bm.getOrCreateAllocatorForIndex(_index_id, total_pages_needed);

    AllocGuard<DiskANNMetadataPage> meta_guard(_allocator);
    _metadata_pid = meta_guard.pid;  // Already a global PID from AllocGuard
    meta_guard->dirty = false;
    meta_guard->max_elements = _max_elements;
    meta_guard->node_count = 0;
    meta_guard->alloc_count.store(1, std::memory_order_relaxed);
    meta_guard->base_pid = -1;
    meta_guard->is_layout_optimized = false;

    // Pre-allocate data pages
    if (num_pages > 0) {
        AllocGuard<VamanaPage> first_page_guard(_allocator);
        _base_pid = first_page_guard.pid;  // Already a global PID
        meta_guard->base_pid = _base_pid;
        first_page_guard->dirty = false;
        first_page_guard->node_count_in_page = 0;
        for (uint64_t i = 1; i < num_pages; ++i) {
            AllocGuard<VamanaPage> page_guard(_allocator);
            page_guard->dirty = false;
            page_guard->node_count_in_page = 0;
        }
    }
    meta_guard->dirty = true;

    _visited_list_pool = std::make_unique<VisitedListPool>(1, _max_elements);
    if (_is_dynamic) _internal_to_external_map.reserve(_max_elements);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::~DiskANN() {}

// ========================================================================
// BATCH DATA ACCESS HELPERS (OPTIMIZED)
// ========================================================================

// --- Helper methods to get or create thread pools ---
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
ThreadPool* DiskANN<T, TagT, Dim, MaxTagsPerNode>::getOrCreateSearchPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(search_pool_mutex_);
    if (!search_thread_pool_ || search_pool_size_ != num_threads) {
        search_thread_pool_.reset(new ThreadPool(num_threads));
        search_pool_size_ = num_threads;
    }
    return search_thread_pool_.get();
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
ThreadPool* DiskANN<T, TagT, Dim, MaxTagsPerNode>::getOrCreateAddPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(add_pool_mutex_);
    if (!add_thread_pool_ || add_pool_size_ != num_threads) {
        add_thread_pool_.reset(new ThreadPool(num_threads));
        add_pool_size_ = num_threads;
    }
    return add_thread_pool_.get();
}

// Local helper for PID grouping without tree/map.
namespace {
struct IdWithIdx {
    PID pid;
    uint32_t id;
    size_t out_idx;
};
inline uint32_t idxInPage(uint32_t id, uint32_t nodesPerPage) { return id % nodesPerPage; }
}  // namespace

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::get_nodes_data_batch(const std::vector<uint32_t>& ids,
                                                                 NodeDataBatch& data_batch) const {
    if (ids.empty()) return true;

    // Build linear items and sort by PID to cluster page accesses.
    thread_local std::vector<IdWithIdx> items;
    items.clear();
    items.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        items.push_back({getNodePID(ids[i]), ids[i], i});
    }
    std::sort(items.begin(), items.end(), [](const IdWithIdx& a, const IdWithIdx& b) { return a.pid < b.pid; });

    // Prefetch unique PIDs in order.
    thread_local std::vector<PID> pids;
    pids.clear();
    pids.reserve(items.size());
    for (size_t i = 0; i < items.size();) {
        PID p = items[i].pid;
        pids.push_back(p);
        while (i < items.size() && items[i].pid == p) ++i;
    }
    if (!pids.empty()) bm.prefetchPages(pids.data(), pids.size());

    // Prepare output buffers.
    data_batch.clear();
    data_batch.vectors.resize(ids.size() * Dim);
    data_batch.tag_counts.resize(ids.size());
    data_batch.tags.clear();
    data_batch.tags.reserve(ids.size());  // will grow as needed

    try {
        size_t i = 0;
        while (i < items.size()) {
            PID p = items[i].pid;
            GuardORelaxed<VamanaPage> g(p);
            // Consume this page's nodes
            for (; i < items.size() && items[i].pid == p; ++i) {
                const auto node_id = items[i].id;
                const auto out_idx = items[i].out_idx;
                NodeAccessor acc(g.ptr, idxInPage(node_id, _NodesPerPage), this);

                // Copy vector
                memcpy(data_batch.vectors.data() + out_idx * Dim, acc.getVector(), VectorSize);

                // Copy tags (append into flat tags, record count per slot)
                auto tags_span = acc.getTags();
                data_batch.tag_counts[out_idx] = static_cast<uint16_t>(tags_span.size());
                data_batch.tags.insert(data_batch.tags.end(), tags_span.begin(), tags_span.end());
            }
        }
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::get_vectors_batch(const std::vector<uint32_t>& ids,
                                                              T* dest_buffer) const {
    if (ids.empty()) return true;

    try {
        // Process IDs in order without sorting or prefetching
        for (size_t i = 0; i < ids.size(); ++i) {
            const uint32_t node_id = ids[i];
            GuardORelaxed<VamanaPage> g(getNodePID(node_id));
            NodeAccessor acc(g.ptr, idxInPage(node_id, _NodesPerPage), this);
            // Optimized copy using AVX2 for better performance
            const float* src = acc.getVector();
            float* dst = dest_buffer + i * Dim;
            constexpr size_t bytes = VectorSize;
            
            #if defined(__AVX2__) && !defined(CALIBY_NO_AVX2)
            // Use AVX2 for faster copy (256-bit = 32 bytes = 8 floats at a time)
            if constexpr (bytes >= 32) {
                size_t avx_iterations = bytes / 32;
                for (size_t j = 0; j < avx_iterations; ++j) {
                    __m256 v = _mm256_loadu_ps(src + j * 8);
                    _mm256_storeu_ps(dst + j * 8, v);
                }
                // Handle remainder
                size_t remainder = bytes % 32;
                if (remainder > 0) {
                    memcpy(dst + (avx_iterations * 8), src + (avx_iterations * 8), remainder);
                }
            } else {
                memcpy(dst, src, bytes);
            }
            #else
            memcpy(dst, src, bytes);
            #endif
        }
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}

// ========================================================================
// DIRECT DISTANCE COMPUTATION (NO COPY)
// ========================================================================
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::compute_distances_batch(
    const std::vector<uint32_t>& ids,
    const T* query,
    std::vector<std::pair<float, uint32_t>>& distances_out) const {
    if (ids.empty()) return true;

    distances_out.clear();
    distances_out.reserve(ids.size());

    // Collect PIDs and offsets without deduplication for prefetching
    // thread_local std::vector<PID> pids;
    // thread_local std::vector<u32> offsets;
    // pids.clear();
    // offsets.clear();
    // pids.reserve(ids.size());
    // offsets.reserve(ids.size());
    
    // for (uint32_t node_id : ids) {
    //     pids.push_back(getNodePID(node_id));
    //     offsets.push_back(static_cast<u32>(idxInPage(node_id, _NodesPerPage) * _FixedNodeSize));
    // }
    
    // // Prefetch all pages with vector offsets (no deduplication)
    // if (!pids.empty()) {
    //     bm.prefetchPages(pids.data(), pids.size(), offsets.data());
    // }

    try {
        // Process IDs in order
        for (uint32_t node_id : ids) {
            GuardORelaxed<VamanaPage> g(getNodePID(node_id));
            NodeAccessor acc(g.ptr, idxInPage(node_id, _NodesPerPage), this);
            const T* vector = acc.getVector();
            float dist = distance_metric.compare(query, vector, Dim);
            distances_out.emplace_back(dist, node_id);
        }
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}

// ========================================================================
// SEARCH ALGORITHMS (OPTIMIZED)
// ========================================================================
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::greedy_search_final(
    uint32_t start_node_id, const T* start_node_vec, const T* query, size_t K, const SearchParams& params,
    const TagT* filter_label) {
    alignas(32) T aligned_query[Dim];
    memcpy(aligned_query, query, VectorSize);

    VisitedList* visited_nodes = _visited_list_pool->getFreeVisitedList();
    auto visited_guard = std::unique_ptr<VisitedList, std::function<void(VisitedList*)>>(
        visited_nodes, [&](VisitedList* p) { _visited_list_pool->releaseVisitedList(p); });

    thread_local static NeighborPriorityQueue best_L_nodes;
    best_L_nodes.clear();
    best_L_nodes.reserve(params.L_search);

    // Seed from provided vector (cache hit) or disk
    if (visited_nodes->try_visit(start_node_id)) {
        if (start_node_vec) {
            float dist = distance_metric.compare(aligned_query, start_node_vec, Dim);
            best_L_nodes.insert(Neighbor(start_node_id, dist));
        } else {
            alignas(32) T vec_from_disk[Dim];
            if (getNodeVector(start_node_id, vec_from_disk)) {
                float dist = distance_metric.compare(aligned_query, vec_from_disk, Dim);
                best_L_nodes.insert(Neighbor(start_node_id, dist));
            } else {
                visited_nodes->unvisit(start_node_id);
            }
        }
    }

    // Beam expand
    std::vector<uint32_t> expansion_candidate_ids;
    std::vector<uint32_t> all_neighbors_from_beam;
    std::vector<uint32_t> unvisited_neighbors_ids;
    // std::vector<PID> pids_to_prefetch;

    while (best_L_nodes.has_unexpanded_node()) {
        expansion_candidate_ids.clear();
        for (size_t i = 0; i < params.beam_width && best_L_nodes.has_unexpanded_node(); ++i) {
            Neighbor candidate = best_L_nodes.closest_unexpanded();
            expansion_candidate_ids.push_back(candidate.id);
        }

        all_neighbors_from_beam.clear();
        for (uint32_t candidate_id : expansion_candidate_ids) {
            std::vector<uint32_t> neighbors;
            if (getNodeNeighbors(candidate_id, neighbors)) {
                all_neighbors_from_beam.insert(all_neighbors_from_beam.end(), neighbors.begin(), neighbors.end());
            }
        }

        unvisited_neighbors_ids.clear();
        // pids_to_prefetch.clear();
        // Prefetch pages for unvisited neighbors before computing distances

        unvisited_neighbors_ids.reserve(all_neighbors_from_beam.size());
        for (uint32_t neighbor_id : all_neighbors_from_beam) {
            if (visited_nodes->try_visit(neighbor_id)) {
                unvisited_neighbors_ids.push_back(neighbor_id);
                // pids_to_prefetch.push_back(getNodePID(neighbor_id));
            }
        }
        if (unvisited_neighbors_ids.empty()) continue;
        // if (!pids_to_prefetch.empty()) {
        //     bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
        // }

        if (filter_label) {
            thread_local static NodeDataBatch data_batch;
            if (!get_nodes_data_batch(unvisited_neighbors_ids, data_batch)) {
                for (uint32_t id : unvisited_neighbors_ids) visited_nodes->unvisit(id);
                continue;
            }
            size_t tags_offset = 0;
            for (size_t i = 0; i < unvisited_neighbors_ids.size(); ++i) {
                uint16_t tag_count = data_batch.tag_counts[i];
                bool has_label = false;
                std::span<const TagT> current_tags(data_batch.tags.data() + tags_offset, tag_count);
                for (const TagT& tag : current_tags) {
                    if (tag == *filter_label) {
                        has_label = true;
                        break;
                    }
                }
                tags_offset += tag_count;
                if (has_label) {
                    const T* vector = data_batch.vectors.data() + i * Dim;
                    float dist = distance_metric.compare(aligned_query, vector, Dim);
                    best_L_nodes.insert(Neighbor(unvisited_neighbors_ids[i], dist));
                }
            }
        } else {
            // Compute distances directly without copying vectors
            thread_local static std::vector<std::pair<float, uint32_t>> distances;
            if (!compute_distances_batch(unvisited_neighbors_ids, aligned_query, distances)) {
                for (uint32_t id : unvisited_neighbors_ids) visited_nodes->unvisit(id);
                continue;
            }
            for (const auto& [dist, id] : distances) {
                best_L_nodes.insert(Neighbor(id, dist));
            }
        }
    }

    std::vector<std::pair<float, uint32_t>> results;
    size_t result_count = std::min((size_t)K, best_L_nodes.size());
    results.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        results.emplace_back(best_L_nodes[i].distance, best_L_nodes[i].id);
    }
    return results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::greedy_search_original(
    const std::vector<uint32_t>& start_nodes, const T* query, size_t K, const SearchParams& params,
    const TagT* filter_label) {
    alignas(32) T aligned_query[Dim];
    memcpy(aligned_query, query, VectorSize);

    VisitedList* visited_nodes = _visited_list_pool->getFreeVisitedList();
    auto visited_guard = std::unique_ptr<VisitedList, std::function<void(VisitedList*)>>(
        visited_nodes, [&](VisitedList* p) { _visited_list_pool->releaseVisitedList(p); });

    thread_local static NeighborPriorityQueue best_L_nodes;
    best_L_nodes.clear();
    best_L_nodes.reserve(params.L_search);

    for (uint32_t start_node : start_nodes) {
        if (visited_nodes->try_visit(start_node)) {
            alignas(32) T start_node_vec[Dim];
            if (getNodeVector(start_node, start_node_vec)) {
                float dist = distance_metric.compare(aligned_query, start_node_vec, Dim);
                best_L_nodes.insert(Neighbor(start_node, dist));
            } else {
                visited_nodes->unvisit(start_node);
            }
        }
    }

    std::vector<uint32_t> unvisited_neighbors_ids;
    //std::vector<PID> pids_to_prefetch;
    //std::vector<u32> offsets_to_prefetch;

    while (best_L_nodes.has_unexpanded_node()) {
        Neighbor candidate = best_L_nodes.closest_unexpanded();

        std::vector<uint32_t> neighbors;
        if (!getNodeNeighbors(candidate.id, neighbors)) continue;

        unvisited_neighbors_ids.clear();
        for (uint32_t neighbor_id : neighbors) {
            if (visited_nodes->try_visit(neighbor_id)) {
                unvisited_neighbors_ids.push_back(neighbor_id);
            }
        }
        if (unvisited_neighbors_ids.empty()) continue;

        // Prefetch pages for unvisited neighbors before computing distances
        // pids_to_prefetch.clear();
        // pids_to_prefetch.reserve(unvisited_neighbors_ids.size());
        // for (uint32_t neighbor_id : unvisited_neighbors_ids) {
        //     pids_to_prefetch.push_back(getNodePID(neighbor_id));
        //     //offsets_to_prefetch.push_back(static_cast<u32>(idxInPage(neighbor_id, _NodesPerPage) * _FixedNodeSize));
        // }
        // if (!pids_to_prefetch.empty()) {
        //     bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size());
        //     //bm.prefetchPages(pids_to_prefetch.data(), pids_to_prefetch.size(), offsets_to_prefetch.data());
        // }

        // Compute distances directly without copying vectors
        thread_local static std::vector<std::pair<float, uint32_t>> distances;
        if (!compute_distances_batch(unvisited_neighbors_ids, aligned_query, distances)) {
            for (uint32_t id : unvisited_neighbors_ids) visited_nodes->unvisit(id);
            continue;
        }

        for (const auto& [dist, id] : distances) {
            best_L_nodes.insert(Neighbor(id, dist));
        }
    }

    std::vector<std::pair<float, uint32_t>> results;
    size_t result_count = std::min((size_t)params.L_search, best_L_nodes.size());
    results.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        results.emplace_back(best_L_nodes[i].distance, best_L_nodes[i].id);
    }
    return results;
}

// ========================================================================
// BUILD PROCESS & OTHER FUNCTIONS
// ========================================================================

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::build_typed(const T* data, const std::vector<std::vector<TagT>>& tags,
                                                        uint64_t num_points, const BuildParams& params) {
    if (num_points > _max_elements) throw std::runtime_error("Number of points exceeds max_elements capacity.");
    CALIBY_LOG_INFO("DiskANN", "Starting index build for ", num_points, " points with R=", _R_max_degree, "...");

    // Phase 1: write vectors & tags
    CALIBY_LOG_INFO("DiskANN", "Phase 1/4: Writing vector and tag data...");
    if (!_is_dynamic) {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        _internal_to_external_map.resize(num_points);
        std::iota(_internal_to_external_map.begin(), _internal_to_external_map.end(), 0);
        _external_to_internal_map.reserve(num_points);
        for (uint32_t i = 0; i < num_points; ++i) _external_to_internal_map[i] = i;
    }
    for (uint64_t i = 0; i < num_points; ++i) {
        PID pid = getNodePID(static_cast<uint32_t>(i));
        GuardX<VamanaPage> page_guard(pid);
        page_guard->node_count_in_page =
            std::max(page_guard->node_count_in_page, static_cast<uint16_t>(getNodeIndexInPage(static_cast<uint32_t>(i)) + 1));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(static_cast<uint32_t>(i)), this);
        acc.setVector(data + i * Dim);
        acc.setTags(tags[i]);
    }
    {
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        meta_guard->node_count = num_points;
        meta_guard->is_layout_optimized = false;
    }
    _is_layout_optimized = false;

    // Phase 2: medoids
    CALIBY_LOG_INFO("DiskANN", "Phase 2/4: Computing medoids...");
    compute_medoids(num_points);

    // Phase 3: random graph init (no hash allocations)
    CALIBY_LOG_INFO("DiskANN", "Phase 3/4: Initializing random graph...");
    std::mt19937 rng(1001);
    for (uint32_t i = 0; i < num_points; ++i) {
        std::vector<uint32_t> neighbors;
        neighbors.reserve(_R_max_degree);
        VisitedList* vis = _visited_list_pool->getFreeVisitedList();
        vis->try_visit(i);  // avoid self
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(num_points - 1));
        while (neighbors.size() < _R_max_degree && neighbors.size() < num_points - 1) {
            uint32_t cand = dist(rng);
            if (vis->try_visit(cand)) {
                if (cand != i) neighbors.push_back(cand);
            }
        }
        _visited_list_pool->releaseVisitedList(vis);

        PID pid = getNodePID(i);
        GuardX<VamanaPage> page_guard(pid);
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(i), this);
        acc.setNeighbors(neighbors);
    }

    // Phase 4: Vamana refinement
    CALIBY_LOG_INFO("DiskANN", "Phase 4/4: Running Vamana refinement passes...");
    std::vector<uint32_t> permutation(num_points);
    std::iota(permutation.begin(), permutation.end(), 0);

    auto run_pass = [&](float alpha, bool filter_aware, int pass_num) {
        CALIBY_LOG_INFO("DiskANN", "  - Pass ", pass_num, " with alpha=", alpha, "...");
        std::shuffle(permutation.begin(), permutation.end(), rng);

        std::atomic<size_t> processed_count{0};
        size_t num_threads = (params.num_threads == 0) ? omp_get_max_threads() : params.num_threads;
        omp_set_num_threads(static_cast<int>(num_threads));

#pragma omp parallel for schedule(static)  // static keeps better locality and avoids scheduler contention
        for (size_t i = 0; i < num_points; ++i) {
            const uint32_t node_id = permutation[i];

            // Per-thread reusable buffers
            alignas(32) T query_vec[Dim];
            std::vector<TagT> node_tags;
            std::vector<uint32_t> start_nodes;

            for (;;) {
                try {
                    if (!getNodeVector(node_id, query_vec)) throw OLCRestartException();
                    if (filter_aware) {
                        node_tags.clear();
                        if (!getNodeTags(node_id, node_tags)) throw OLCRestartException();
                        start_nodes.clear();
                        for (const TagT& tag : node_tags) {
                            auto it = _medoids.find(tag);
                            if (it != _medoids.end()) start_nodes.push_back(it->second);
                        }
                    } else {
                        start_nodes.clear();
                    }

                    if (start_nodes.empty() && !_medoids.empty()) start_nodes.push_back(_medoids.begin()->second);
                    if (start_nodes.empty()) start_nodes.push_back(0);

                    SearchParams search_params{params.L_build, 4};
                    auto candidates_with_dist =
                        greedy_search_original(start_nodes, query_vec, params.L_build, search_params, nullptr);

                    robust_prune_batched(node_id, candidates_with_dist, alpha, filter_aware);

                    std::vector<uint32_t> final_neighbors;
                    final_neighbors.reserve(candidates_with_dist.size());
                    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);

                    connect_neighbors_batched(node_id, final_neighbors, params, filter_aware);
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }

            size_t count = processed_count.fetch_add(1) + 1;
            if ((count % (num_points / 100 + 1)) == 0) {
                CALIBY_LOG_DEBUG("DiskANN", "    Processed ", count, "/", num_points, " nodes.");
            }
        }
    };

    run_pass(1.0f, true, 1);
    run_pass(params.alpha, true, 2);

    CALIBY_LOG_INFO("DiskANN", "Build complete.");

    optimize_layout();
    CALIBY_LOG_INFO("DiskANN", "Layout optimization finished. Index is ready for search.");
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::robust_prune_batched(uint32_t /*p_id*/,
                                                                 std::vector<std::pair<float, uint32_t>>& candidates,
                                                                 float alpha, bool /*filter_aware*/) {
    if (candidates.empty()) return;

    // Keep only top alpha*R by distance (partial-select), then do linear pruning with early stop.
    const size_t R = _R_max_degree;
    const size_t alphaR = std::min(candidates.size(), (size_t)std::ceil(alpha * R));
    std::nth_element(candidates.begin(), candidates.begin() + alphaR, candidates.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    candidates.resize(alphaR);
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Batch fetch vectors of the reduced candidate set.
    thread_local static std::vector<uint32_t> ids;
    ids.clear();
    ids.reserve(candidates.size());
    for (auto& p : candidates) ids.push_back(p.second);

    thread_local static std::vector<T, AlignedAllocator<T, 32>> cand_vecs;
    cand_vecs.resize(ids.size() * Dim);
    if (!get_vectors_batch(ids, cand_vecs.data())) {
        if (candidates.size() > R) candidates.resize(R);
        return;
    }

    std::vector<std::pair<float, uint32_t>> out;
    out.reserve(R);
    std::vector<uint8_t> pruned(candidates.size(), 0);

    for (size_t i = 0; i < candidates.size() && out.size() < R; ++i) {
        if (pruned[i]) continue;
        out.push_back(candidates[i]);
        const T* vi = cand_vecs.data() + i * Dim;

        // Local triangle pruning within the reduced window; early exit if R reached.
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (pruned[j]) continue;
            const T* vj = cand_vecs.data() + j * Dim;
            float dij = distance_metric.compare(vi, vj, Dim);
            if (alpha * dij < candidates[j].first) pruned[j] = 1;
        }
    }

    candidates.swap(out);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::connect_neighbors_batched(uint32_t node_id,
                                                                      const std::vector<uint32_t>& neighbors,
                                                                      const BuildParams& params, bool filter_aware) {
    // Write N(node_id) once.
    {
        GuardX<VamanaPage> page_guard(getNodePID(node_id));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        acc.setNeighbors(neighbors);
    }

    // For each v in N(node_id), ensure symmetry v <- node_id with bounded pruning if needed.
    // We keep locking narrow and only recompute when degree overflow occurs.
    for (uint32_t v : neighbors) {
        for (;;) {
            try {
                std::vector<uint32_t> nbors_v;
                if (!getNodeNeighbors(v, nbors_v)) throw OLCRestartException();

                // Quick check: already connected?
                bool connected = false;
                for (uint32_t u : nbors_v) {
                    if (u == node_id) {
                        connected = true;
                        break;
                    }
                }
                if (!connected) {
                    nbors_v.push_back(node_id);
                    if (nbors_v.size() > _R_max_degree) {
                        // Bounded prune on v
                        thread_local static std::vector<T, AlignedAllocator<T, 32>> vecs;
                        vecs.resize(nbors_v.size() * Dim);
                        if (!get_vectors_batch(nbors_v, vecs.data())) throw OLCRestartException();

                        alignas(32) T v_query[Dim];
                        if (!getNodeVector(v, v_query)) throw OLCRestartException();

                        std::vector<std::pair<float, uint32_t>> cand;
                        cand.reserve(nbors_v.size());
                        for (size_t i = 0; i < nbors_v.size(); ++i) {
                            float d = distance_metric.compare(v_query, vecs.data() + i * Dim, Dim);
                            cand.emplace_back(d, nbors_v[i]);
                        }
                        robust_prune_batched(v, cand, params.alpha, filter_aware);
                        nbors_v.clear();
                        nbors_v.reserve(cand.size());
                        for (auto& p : cand) nbors_v.push_back(p.second);
                    }

                    // Commit N(v)
                    {
                        GuardX<VamanaPage> xg(getNodePID(v));
                        MutableNodeAccessor xacc(xg.ptr, getNodeIndexInPage(v), this);
                        xacc.setNeighbors(nbors_v);
                    }
                }
                break;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
    }
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::optimize_layout() {
    if (_is_dynamic) {
        CALIBY_LOG_WARN("DiskANN", "Layout optimization is intended for static indices...");
    }

    uint64_t num_points = 0;
    for (;;) {
        try {
            GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
            if (meta_guard->is_layout_optimized) {
                CALIBY_LOG_INFO("DiskANN", "Layout is already optimized. Skipping.");
                _is_layout_optimized = true;
                return;
            }
            num_points = meta_guard->node_count;
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
    if (num_points == 0) return;

    CALIBY_LOG_INFO("DiskANN", "Starting graph layout optimization for ", num_points, " points...");

    // Phase 1: BFS ordering
    CALIBY_LOG_INFO("DiskANN", "Phase 1/4: Calculating BFS ordering...");
    std::vector<uint32_t> new_id_to_old_id(num_points);
    std::vector<int32_t> old_id_to_new_id(num_points, -1);
    std::vector<bool> visited(num_points, false);
    std::queue<uint32_t> q;
    uint32_t start_node = _medoids.empty() ? 0 : _medoids.begin()->second;
    q.push(start_node);
    visited[start_node] = true;
    uint32_t current_new_id = 0;

    while (current_new_id < num_points) {
        if (q.empty()) {
            bool found_unvisited = false;
            for (uint32_t i = 0; i < num_points; ++i) {
                if (!visited[i]) {
                    q.push(i);
                    visited[i] = true;
                    found_unvisited = true;
                    break;
                }
            }
            if (!found_unvisited) break;
        }
        uint32_t current_old_id = q.front();
        q.pop();
        new_id_to_old_id[current_new_id] = current_old_id;
        old_id_to_new_id[current_old_id] = current_new_id;
        current_new_id++;

        std::vector<uint32_t> neighbors;
        for (;;) {
            try {
                if (!getNodeNeighbors(current_old_id, neighbors)) throw OLCRestartException();
                break;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
        for (uint32_t neighbor_old_id : neighbors) {
            if (!visited[neighbor_old_id]) {
                visited[neighbor_old_id] = true;
                q.push(neighbor_old_id);
            }
        }
    }

    // Phase 2: read original node blobs
    CALIBY_LOG_INFO("DiskANN", "Phase 2/4: Reading original index data into memory snapshot...");
    std::vector<uint8_t> original_data_buffer(num_points * _FixedNodeSize);
    for (uint32_t old_id = 0; old_id < num_points; ++old_id) {
        for (;;) {
            try {
                GuardO<VamanaPage> page_guard(getNodePID(old_id));
                const uint8_t* node_data_ptr = page_guard->getNodeData() + getNodeIndexInPage(old_id) * _FixedNodeSize;
                memcpy(original_data_buffer.data() + old_id * _FixedNodeSize, node_data_ptr, _FixedNodeSize);
                break;
            } catch (const OLCRestartException&) {
                continue;
            }
        }
    }

    // Phase 3: remap and write back
    CALIBY_LOG_INFO("DiskANN", "Phase 3/4: Constructing and writing final index...");
    std::vector<uint8_t> temp_node_buffer(_FixedNodeSize);
    for (uint32_t new_id = 0; new_id < num_points; ++new_id) {
        uint32_t old_id_source = new_id_to_old_id[new_id];
        const uint8_t* src_node_ptr = original_data_buffer.data() + old_id_source * _FixedNodeSize;

        memcpy(temp_node_buffer.data(), src_node_ptr, VectorSize + TagsHeaderSize + TagsDataSize);

        const uint16_t* old_count_ptr =
            reinterpret_cast<const uint16_t*>(src_node_ptr + VectorSize + TagsHeaderSize + TagsDataSize);
        const uint32_t* old_neighbors_data_ptr = reinterpret_cast<const uint32_t*>(old_count_ptr + 1);
        const uint16_t neighbor_count = *old_count_ptr;

        uint16_t* new_count_ptr =
            reinterpret_cast<uint16_t*>(temp_node_buffer.data() + VectorSize + TagsHeaderSize + TagsDataSize);
        uint32_t* new_neighbors_data_ptr = reinterpret_cast<uint32_t*>(new_count_ptr + 1);
        *new_count_ptr = neighbor_count;

        for (uint16_t i = 0; i < neighbor_count; ++i) {
            uint32_t old_neighbor_id = old_neighbors_data_ptr[i];
            new_neighbors_data_ptr[i] =
                (old_neighbor_id < num_points) ? static_cast<uint32_t>(old_id_to_new_id[old_neighbor_id]) : old_neighbor_id;
        }

        GuardX<VamanaPage> page_guard(getNodePID(new_id));
        uint8_t* dest_on_disk_ptr = page_guard->getNodeData() + getNodeIndexInPage(new_id) * _FixedNodeSize;
        memcpy(dest_on_disk_ptr, temp_node_buffer.data(), _FixedNodeSize);
    }

    // Phase 4: update maps
    CALIBY_LOG_INFO("DiskANN", "Phase 4/4: Updating internal ID maps...");
    {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        std::vector<uint32_t> new_internal_to_external_map(num_points);
        for (uint32_t new_id = 0; new_id < num_points; ++new_id) {
            uint32_t old_id = new_id_to_old_id[new_id];
            new_internal_to_external_map[new_id] = _internal_to_external_map[old_id];
        }
        _internal_to_external_map = std::move(new_internal_to_external_map);

        _external_to_internal_map.clear();
        _external_to_internal_map.reserve(num_points);
        for (uint32_t new_id = 0; new_id < num_points; ++new_id) {
            _external_to_internal_map[_internal_to_external_map[new_id]] = new_id;
        }
    }
    for (auto& pair : _medoids) {
        if (pair.second < num_points) pair.second = static_cast<uint32_t>(old_id_to_new_id[pair.second]);
    }

    // Refresh medoid caches after remap
    compute_medoids(num_points);

    {
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        meta_guard->is_layout_optimized = true;
    }
    _is_layout_optimized = true;
    CALIBY_LOG_INFO("DiskANN", "Layout optimization complete.");
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::build(const void* data, const std::vector<std::vector<uint32_t>>& tags,
                                                  uint64_t num_points, const BuildParams& params) {
    build_typed(static_cast<const T*>(data), tags, num_points, params);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search(const void* query, size_t K,
                                                                                      const SearchParams& params) {
    return search_typed(static_cast<const T*>(query), K, params);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_typed(
    const T* query, size_t K, const SearchParams& params) {
    uint32_t start_node_id;
    const T* start_node_vec = nullptr;
    for (;;) {
        try {
            GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
            if (meta_guard->node_count == 0) return {};
            if (_medoids.empty()) compute_medoids(meta_guard->node_count);

            auto it = _medoids.find(0);
            if (it == _medoids.end() && !_medoids.empty()) it = _medoids.begin();
            if (it == _medoids.end()) return {};

            start_node_id = it->second;
            auto cache_it = _medoid_vectors_cache.find(it->first);
            if (cache_it != _medoid_vectors_cache.end()) {
                start_node_vec = cache_it->second.data();
            }
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }
    auto internal_results = greedy_search_final(start_node_id, start_node_vec, query, K, params, nullptr);
    for (auto& pair : internal_results) {
        if (pair.second < _internal_to_external_map.size()) {
            pair.second = _internal_to_external_map[pair.second];
        }
    }
    return internal_results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter(
    const void* query, uint32_t filter_label, size_t K, const SearchParams& params) {
    return search_with_filter_typed(static_cast<const T*>(query), filter_label, K, params);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::pair<float, uint32_t>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_typed(
    const T* query, const TagT& filter_label, size_t K, const SearchParams& params) {
    uint32_t start_node_id;
    const T* start_node_vec = nullptr;

    auto it = _medoids.find(filter_label);
    if (it == _medoids.end()) {
        it = _medoids.find(0);
        if (it == _medoids.end() && !_medoids.empty()) it = _medoids.begin();
        if (it == _medoids.end()) return {};
    }
    start_node_id = it->second;
    auto cache_it = _medoid_vectors_cache.find(it->first);
    if (cache_it != _medoid_vectors_cache.end()) {
        start_node_vec = cache_it->second.data();
    }
    auto internal_results = greedy_search_final(start_node_id, start_node_vec, query, K, params, &filter_label);
    for (auto& pair : internal_results) {
        if (pair.second < _internal_to_external_map.size()) {
            pair.second = _internal_to_external_map[pair.second];
        }
    }
    return internal_results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeVector(uint32_t node_id, T* dest_vector) const {
    try {
        GuardORelaxed<VamanaPage> page_guard(getNodePID(node_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        memcpy(dest_vector, acc.getVector(), VectorSize);
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeTags(uint32_t node_id, std::vector<TagT>& dest_tags) const {
    try {
        GuardORelaxed<VamanaPage> page_guard(getNodePID(node_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        auto tags_span = acc.getTags();
        dest_tags.assign(tags_span.begin(), tags_span.end());
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
bool DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeNeighbors(uint32_t node_id,
                                                             std::vector<uint32_t>& dest_neighbors) const {
    try {
        GuardORelaxed<VamanaPage> page_guard(getNodePID(node_id));
        NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
        auto neighbors_span = acc.getNeighbors();
        dest_neighbors.assign(neighbors_span.begin(), neighbors_span.end());
        return true;
    } catch (const OLCRestartException&) {
        return false;
    }
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
inline PID DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodePID(uint32_t node_id) const {
    return _base_pid + (node_id / _NodesPerPage);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
inline uint32_t DiskANN<T, TagT, Dim, MaxTagsPerNode>::getNodeIndexInPage(uint32_t node_id) const {
    return node_id % _NodesPerPage;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::compute_medoids(uint64_t num_points) {
    if (num_points == 0) return;
    CALIBY_LOG_INFO("DiskANN", "Computing medoids using centroid method...");

    _medoids.clear();
    _medoid_vectors_cache.clear();

    std::unordered_map<TagT, std::vector<uint32_t>> points_by_tag;
    for (uint32_t i = 0; i < num_points; ++i) {
        std::vector<TagT> tags;
        if (getNodeTags(i, tags)) {
            for (const auto& tag : tags) points_by_tag[tag].push_back(i);
        }
    }

    // Universal tag
    TagT universal_tag_key = std::numeric_limits<TagT>::max();
    points_by_tag[universal_tag_key].reserve(num_points);
    for (uint32_t i = 0; i < num_points; ++i) points_by_tag[universal_tag_key].push_back(i);

    // Compute centroid & select closest point per tag
    for (const auto& kv : points_by_tag) {
        const TagT tag = kv.first;
        const auto& points = kv.second;
        if (points.empty()) continue;

        std::vector<double> centroid(Dim, 0.0);
        size_t count = 0;

        const size_t BATCH = 1024;
        thread_local static std::vector<T, AlignedAllocator<T, 32>> vec_batch;
        thread_local static std::vector<uint32_t> id_batch;
        vec_batch.resize(BATCH * Dim);
        id_batch.clear();
        id_batch.reserve(BATCH);

        auto flush_acc = [&]() {
            if (id_batch.empty()) return;
            if (get_vectors_batch(id_batch, vec_batch.data())) {
                for (size_t i = 0; i < id_batch.size(); ++i) {
                    const T* v = vec_batch.data() + i * Dim;
                    for (size_t d = 0; d < Dim; ++d) centroid[d] += v[d];
                }
                count += id_batch.size();
            }
            id_batch.clear();
        };

        for (uint32_t pid : points) {
            id_batch.push_back(pid);
            if (id_batch.size() == BATCH) flush_acc();
        }
        flush_acc();

        if (count == 0) continue;
        for (size_t d = 0; d < Dim; ++d) centroid[d] /= static_cast<double>(count);

        std::vector<T, AlignedAllocator<T, 32>> centroid_T(Dim);
        for (size_t d = 0; d < Dim; ++d) centroid_T[d] = static_cast<T>(centroid[d]);

        // Select nearest to centroid
        float best = std::numeric_limits<float>::max();
        uint32_t medoid = points[0];

        id_batch.clear();
        auto flush_sel = [&]() {
            if (id_batch.empty()) return;
            if (get_vectors_batch(id_batch, vec_batch.data())) {
                for (size_t i = 0; i < id_batch.size(); ++i) {
                    float dist = distance_metric.compare(centroid_T.data(), vec_batch.data() + i * Dim, Dim);
                    if (dist < best) {
                        best = dist;
                        medoid = id_batch[i];
                    }
                }
            }
            id_batch.clear();
        };

        for (uint32_t pid : points) {
            id_batch.push_back(pid);
            if (id_batch.size() == BATCH) flush_sel();
        }
        flush_sel();

        TagT cache_key;
        if (tag == universal_tag_key) {
            if (_medoids.find(0) == _medoids.end()) {
                _medoids[0] = medoid;
                cache_key = 0;
            } else {
                cache_key = 0;  // keep for cache fill below if missing
            }
        } else {
            _medoids[tag] = medoid;
            cache_key = tag;
        }

        if (_medoids.count(cache_key) && _medoid_vectors_cache.find(cache_key) == _medoid_vectors_cache.end()) {
            _medoid_vectors_cache[cache_key].resize(Dim);
            if (!getNodeVector(_medoids[cache_key], _medoid_vectors_cache[cache_key].data())) {
                _medoid_vectors_cache.erase(cache_key);
                CALIBY_LOG_WARN("DiskANN", "Failed to cache vector for medoid tag ", cache_key);
            }
        }
    }

    if (_medoids.empty() && num_points > 0) {
        _medoids[0] = static_cast<uint32_t>(num_points / 2);
        _medoid_vectors_cache[0].resize(Dim);
        if (!getNodeVector(_medoids[0], _medoid_vectors_cache[0].data())) {
            _medoid_vectors_cache.erase(0);
        }
    }

    CALIBY_LOG_INFO("DiskANN", "Computed ", _medoids.size(), " medoids using centroid method.");
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::insert_point(const void* point, const std::vector<uint32_t>& tags,
                                                         uint32_t external_id) {
    insert_point_typed(static_cast<const T*>(point), tags, external_id);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::lazy_delete(uint32_t external_id) {
    if (!_is_dynamic) throw std::runtime_error("Cannot call lazy_delete on a non-dynamic index.");
    uint32_t internal_id = std::numeric_limits<uint32_t>::max();
    {
        std::shared_lock<std::shared_mutex> lock(_map_lock);
        auto it = _external_to_internal_map.find(external_id);
        if (it != _external_to_internal_map.end()) internal_id = it->second;
    }
    if (internal_id != std::numeric_limits<uint32_t>::max()) _deleted_nodes.insert(internal_id);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::insert_point_typed(const T* point, const std::vector<TagT>& tags,
                                                               uint32_t external_id) {
    if (!_is_dynamic) throw std::runtime_error("Cannot call insert_point on a non-dynamic index.");
    if (_is_layout_optimized)
        CALIBY_LOG_WARN("DiskANN", "Inserting into an optimized-layout index will degrade performance over time.");

    uint32_t internal_id;
    {
        std::unique_lock<std::shared_mutex> lock(_map_lock);
        GuardX<DiskANNMetadataPage> meta_guard(_metadata_pid);
        internal_id = meta_guard->node_count.fetch_add(1);
        if (internal_id >= _max_elements) {
            meta_guard->node_count.fetch_sub(1);
            throw std::runtime_error("DiskANN is full.");
        }
        _external_to_internal_map[external_id] = internal_id;
        _internal_to_external_map.push_back(external_id);

        PID pid = getNodePID(internal_id);
        GuardX<VamanaPage> page_guard(pid);
        page_guard->node_count_in_page =
            std::max(page_guard->node_count_in_page, static_cast<uint16_t>(getNodeIndexInPage(internal_id) + 1));
        MutableNodeAccessor acc(page_guard.ptr, getNodeIndexInPage(internal_id), this);
        acc.setVector(point);
        acc.setTags(tags);
        acc.setNeighbors({});
    }

    std::vector<uint32_t> start_nodes;
    for (const TagT& tag : tags)
        if (_medoids.count(tag)) start_nodes.push_back(_medoids.at(tag));
    if (start_nodes.empty() && !_medoids.empty()) start_nodes.push_back(_medoids.begin()->second);
    if (start_nodes.empty() && internal_id > 0) start_nodes.push_back(0);
    if (start_nodes.empty()) return;

    BuildParams insert_build_params;
    SearchParams insert_search_params{insert_build_params.L_build, 4};
    auto candidates_with_dist =
        greedy_search_original(start_nodes, point, insert_build_params.L_build, insert_search_params, nullptr);
    robust_prune_batched(internal_id, candidates_with_dist, insert_build_params.alpha, true);

    std::vector<uint32_t> final_neighbors;
    final_neighbors.reserve(candidates_with_dist.size());
    for (const auto& p : candidates_with_dist) final_neighbors.push_back(p.second);

    connect_neighbors_batched(internal_id, final_neighbors, insert_build_params, true);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
void DiskANN<T, TagT, Dim, MaxTagsPerNode>::consolidate_deletes(const BuildParams& /*params*/) {
    throw std::runtime_error("consolidate_deletes not fully implemented in this optimized version yet.");
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_parallel(
    const void* queries, size_t num_queries, size_t K, const SearchParams& params, size_t num_threads) {
    return search_parallel_typed(static_cast<const T*>(queries), num_queries, K, params, num_threads);
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_parallel_typed(
    const T* queries_typed, size_t num_queries, size_t K, const SearchParams& params, size_t num_threads) {
    std::vector<std::vector<std::pair<float, uint32_t>>> all_results(num_queries);
    if (num_queries == 0) return all_results;

    uint32_t num_points = 0;
    for (;;) {
        try {
            GuardO<DiskANNMetadataPage> meta_guard(_metadata_pid);
            if (meta_guard->node_count == 0) return all_results;
            num_points = static_cast<uint32_t>(meta_guard->node_count);
            if (_medoids.empty()) {
                compute_medoids(meta_guard->node_count);
            }
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    const size_t threads_to_use = (num_threads == 0) ? omp_get_max_threads() : num_threads;

    std::vector<uint32_t> entry_points;
    entry_points.reserve(threads_to_use);
    if (!_medoids.empty()) {
        for (const auto& kv : _medoids) {
            if (entry_points.size() == threads_to_use) break;
            entry_points.push_back(kv.second);
        }
    }
    if (entry_points.size() < threads_to_use) {
        std::mt19937 rng(1001);
        std::uniform_int_distribution<uint32_t> dist(0, num_points - 1);
        while (entry_points.size() < threads_to_use) entry_points.push_back(dist(rng));
    }

    std::vector<std::vector<T, AlignedAllocator<T, 32>>> entry_point_vectors(entry_points.size());
    for (size_t i = 0; i < entry_points.size(); ++i) {
        entry_point_vectors[i].resize(Dim);
        getNodeVector(entry_points[i], entry_point_vectors[i].data());
    }

#pragma omp parallel num_threads(threads_to_use)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t nT = static_cast<size_t>(omp_get_num_threads());

        const size_t chunk = (num_queries + nT - 1) / nT;
        const size_t beg = tid * chunk;
        const size_t end = std::min(num_queries, beg + chunk);

        const uint32_t start_node_id = entry_points[tid % entry_points.size()];
        const T* start_node_vec = entry_point_vectors[tid % entry_point_vectors.size()].data();

        for (size_t i = beg; i < end; ++i) {
            auto res = greedy_search_final(start_node_id, start_node_vec, queries_typed + i * Dim, K, params,
                                           /*filter_label=*/nullptr);
            if (!_internal_to_external_map.empty()) {
                for (auto& p : res) {
                    if (p.second < _internal_to_external_map.size()) {
                        p.second = _internal_to_external_map[p.second];
                    }
                }
            }
            all_results[i] = std::move(res);
        }
    }

    return all_results;
}

template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>> DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_parallel(
    const void* queries, size_t num_queries, uint32_t filter_label, size_t K, const SearchParams& params,
    size_t num_threads) {
    return search_with_filter_parallel_typed(static_cast<const T*>(queries), num_queries, filter_label, K, params,
                                             num_threads);
}
template <typename T, typename TagT, size_t Dim, size_t MaxTagsPerNode>
std::vector<std::vector<std::pair<float, uint32_t>>>
DiskANN<T, TagT, Dim, MaxTagsPerNode>::search_with_filter_parallel_typed(const T* queries_typed, size_t num_queries,
                                                                         const TagT& filter_label, size_t K,
                                                                         const SearchParams& params,
                                                                         size_t num_threads) {
    std::vector<std::vector<std::pair<float, uint32_t>>> all_results(num_queries);
    if (num_queries == 0) return all_results;

    uint32_t start_node_id;
    const T* start_node_vec = nullptr;

    auto it = _medoids.find(filter_label);
    if (it == _medoids.end()) {
        it = _medoids.find(0);
        if (it == _medoids.end() && !_medoids.empty()) it = _medoids.begin();
        if (it == _medoids.end()) return {};
    }
    start_node_id = it->second;
    auto cache_it = _medoid_vectors_cache.find(it->first);
    if (cache_it != _medoid_vectors_cache.end()) {
        start_node_vec = cache_it->second.data();
    }

    size_t threads_to_use = (num_threads == 0) ? omp_get_max_threads() : num_threads;
#pragma omp parallel for num_threads(threads_to_use) schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(num_queries); ++i) {
        all_results[static_cast<size_t>(i)] =
            greedy_search_final(start_node_id, start_node_vec, queries_typed + static_cast<size_t>(i) * Dim, K, params,
                                &filter_label);
    }

    if (!_internal_to_external_map.empty()) {
#pragma omp parallel for num_threads(threads_to_use) schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(num_queries); ++i) {
            auto& v = all_results[static_cast<size_t>(i)];
            for (auto& p : v) {
                if (p.second < _internal_to_external_map.size()) {
                    p.second = _internal_to_external_map[p.second];
                }
            }
        }
    }

    return all_results;
}

// Explicit template instantiations
template class DiskANN<float, uint32_t, 16>;
template class DiskANN<float, uint32_t, 32>;
template class DiskANN<float, uint32_t, 64>;
template class DiskANN<float, uint32_t, 96>;
template class DiskANN<float, uint32_t, 128>;
template class DiskANN<float, uint32_t, 256>;
template class DiskANN<float, uint32_t, 384>;
template class DiskANN<float, uint32_t, 512>;
template class DiskANN<float, uint32_t, 768>;
template class DiskANN<float, uint32_t, 1024>;
