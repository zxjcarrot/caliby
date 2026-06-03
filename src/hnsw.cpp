#include "hnsw.hpp"
#include "logging.hpp"

#include <immintrin.h>
#include <cstdio>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>

// --- Custom inline heap (faster than std::priority_queue) ---
namespace {
struct HeapCand { float dist; u32 id; };
inline void push_min_heap(std::vector<HeapCand>& h, HeapCand v) noexcept {
    h.push_back(v); size_t hole = h.size()-1;
    while (hole > 0) { size_t p = (hole-1)>>1; if (h[p].dist <= v.dist) break; h[hole] = h[p]; hole = p; }
    h[hole] = v;
}
inline HeapCand pop_min_heap(std::vector<HeapCand>& h) noexcept {
    HeapCand r = h.front(), v = h.back(); h.pop_back();
    if (!h.empty()) { size_t len = h.size(), hole = 0, child = 1;
        while (child < len) { size_t right = child+1;
            if (right < len && h[right].dist < h[child].dist) child = right;
            if (h[child].dist >= v.dist) break; h[hole] = h[child]; hole = child; child = (hole<<1)+1; }
        h[hole] = v; }
    return r;
}
inline void push_max_heap(std::vector<HeapCand>& h, HeapCand v) noexcept {
    h.push_back(v); size_t hole = h.size()-1;
    while (hole > 0) { size_t p = (hole-1)>>1; if (h[p].dist >= v.dist) break; h[hole] = h[p]; hole = p; }
    h[hole] = v;
}
inline void replace_top_max_heap(std::vector<HeapCand>& h, HeapCand v) noexcept {
    size_t len = h.size(), hole = 0, child = 1;
    while (child < len) { size_t right = child+1;
        if (right < len && h[right].dist > h[child].dist) child = right;
        if (h[child].dist <= v.dist) break; h[hole] = h[child]; hole = child; child = (hole<<1)+1; }
    h[hole] = v;
}
}  // namespace
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

// --- Helper methods to get or create thread pools ---
template <typename DistanceMetric>
ThreadPool* HNSW<DistanceMetric>::getOrCreateSearchPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(search_pool_mutex_);
    if (!search_thread_pool_ || search_pool_size_ != num_threads) {
        search_thread_pool_.reset(new ThreadPool(num_threads));
        search_pool_size_ = num_threads;
    }
    return search_thread_pool_.get();
}

template <typename DistanceMetric>
ThreadPool* HNSW<DistanceMetric>::getOrCreateAddPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(add_pool_mutex_);
    if (!add_thread_pool_ || add_pool_size_ != num_threads) {
        add_thread_pool_.reset(new ThreadPool(num_threads));
        add_pool_size_ = num_threads;
    }
    return add_thread_pool_.get();
}

// --- Helper function to estimate MaxLevel ---
template <typename DistanceMetric>
size_t HNSW<DistanceMetric>::estimateMaxLevel(u64 max_elements, size_t M) {
    // The random level generation uses: level = -log(uniform_random) / log(M)
    // For max_elements, the expected maximum level is approximately log(max_elements) / log(M)
    if (max_elements <= 1) return 1;

    // Correct formula based on the actual random level generation algorithm
    double expected_max_level = std::log(static_cast<double>(max_elements)) / std::log(static_cast<double>(M));
    size_t estimated_max_level = static_cast<size_t>(std::ceil(expected_max_level + 1));  // +2 for safety margin

    // Ensure reasonable bounds: at least 3 levels, at most 16 levels
    return std::max(static_cast<size_t>(3), std::min(static_cast<size_t>(16), estimated_max_level));
}

// --- HNSW Constructor Implementation ---
template <typename DistanceMetric>
HNSW<DistanceMetric>::HNSW(u64 max_elements, size_t dim, size_t M_param, size_t ef_construction_param,
                           bool enable_prefetch_param, bool skip_recovery_param, uint32_t index_id, const std::string& name)
        : index_id_(index_id),
          name_(name),
          Dim(dim),
          M(M_param),
          M0(2 * M_param),  // M0 = 2 * M as requested
          efConstruction(ef_construction_param),
          MaxLevel(estimateMaxLevel(max_elements, M_param)),
          MaxNeighborsHeaderSize(MaxLevel * sizeof(u16)),
          MaxNeighborsListSize((M0 + (MaxLevel - 1) * M) * sizeof(u32)),
          VectorSize(dim * sizeof(float)),
          FixedNodeSize(VectorSize + MaxNeighborsHeaderSize + MaxNeighborsListSize),
          MaxNodesPerPage((pageSize - HNSWPage::HeaderSize) / FixedNodeSize),
          NodesPerPage(MaxNodesPerPage),
          enable_prefetch_(enable_prefetch_param) {
    if (Dim == 0) {
        throw std::runtime_error("HNSW dimension must be greater than zero.");
    }

    CALIBY_LOG_INFO("HNSW", "Initialization: Dim=", Dim, ", M=", M, ", M0=", M0, ", efConstruction=", efConstruction, ", MaxLevel=", MaxLevel, ", FixedNodeSize=", FixedNodeSize, " bytes, NodesPerPage=", MaxNodesPerPage, ", enable_prefetch=", enable_prefetch_);
    // Validate that we can fit at least one node per page
    if (MaxNodesPerPage == 0) {
        throw std::runtime_error(
            "Page size is too small to fit even one node with the given parameters. "
            "Consider reducing M or vector dimension.");
    }
    this->max_elements_ = max_elements;
    mult_factor = 1.0 / std::log(1.0 * M);
    recovered_from_disk_ = false;
    
    // Get or create PIDAllocator for this index
    // Calculate max pages needed: 1 metadata page + data pages + buffer for chunk allocation
    u64 data_pages = (max_elements + NodesPerPage - 1) / NodesPerPage;
    // Add 10% buffer + 2048 (chunk size) for safety margin during construction
    u64 total_pages_needed = 1 + data_pages + (data_pages / 10) + 2048;
    allocator_ = bm.getOrCreateAllocatorForIndex(index_id_, total_pages_needed);

    // Compute metadata page ID - use encoded PID only for Array2Level mode
    PID global_metadata_page_id;
    if (bm.supportsMultiIndexPIDs() && index_id_ > 0) {
        // Multi-index mode: encode index_id in high 32 bits
        global_metadata_page_id = (static_cast<PID>(index_id_) << 32) | 0ULL;
    } else {
        // Single-level mode: use simple sequential PID 0
        global_metadata_page_id = 0;
    }
    
    GuardX<MetaDataPage> meta_page_guard(global_metadata_page_id);
    MetaDataPage* meta_page_ptr = meta_page_guard.ptr;
    HNSWMetaInfo* meta_info = &meta_page_ptr->hnsw_meta;
    const bool has_existing_meta = meta_info->isValid();
    const bool params_match = has_existing_meta &&
                              meta_info->max_elements == max_elements && meta_info->dim == Dim && meta_info->M == M &&
                              meta_info->ef_construction == efConstruction &&
                              meta_info->max_level == MaxLevel;

    CALIBY_LOG_DEBUG("HNSW", "Recovery: skip_recovery=", (skip_recovery_param ? "true" : "false"),
              " has_existing_meta=", (has_existing_meta ? "true" : "false"));
    if (has_existing_meta) {
    CALIBY_LOG_DEBUG("HNSW", "Recovery: stored meta: max_elements=", meta_info->max_elements,
          " dim=", meta_info->dim, " M=", meta_info->M,
          " ef_construction=", meta_info->ef_construction,
          " max_level=", meta_info->max_level,
          " metadata_pid=", meta_info->metadata_pid,
          " base_pid=", meta_info->base_pid,
          " valid_flag=", static_cast<u32>(meta_info->valid));
    }
    CALIBY_LOG_DEBUG("HNSW", "Recovery: params_match=", (params_match ? "true" : "false"));
    if (has_existing_meta && !params_match) {
        if (meta_info->max_elements != max_elements) {
            CALIBY_LOG_DEBUG("HNSW", "Recovery: mismatch: stored max_elements=", meta_info->max_elements,
                      " requested=", max_elements);
        }
        if (meta_info->dim != Dim) {
            CALIBY_LOG_DEBUG("HNSW", "Recovery: mismatch: stored dim=", meta_info->dim, " requested=", Dim);
        }
        if (meta_info->M != M) {
            CALIBY_LOG_DEBUG("HNSW", "Recovery: mismatch: stored M=", meta_info->M, " requested=", M);
        }
        if (meta_info->ef_construction != efConstruction) {
            CALIBY_LOG_DEBUG("HNSW", "Recovery: mismatch: stored ef_construction=", meta_info->ef_construction,
                      " requested=", efConstruction);
        }
        if (meta_info->max_level != MaxLevel) {
            CALIBY_LOG_DEBUG("HNSW", "Recovery: mismatch: stored max_level=", meta_info->max_level,
                      " requested=", MaxLevel);
        }
    }

    bool recovered = false;
    if (!skip_recovery_param && params_match) {
        this->metadata_pid = meta_info->metadata_pid;
        this->base_pid = meta_info->base_pid;
        this->max_elements_ = meta_info->max_elements;
        recovered = true;
        recovered_from_disk_ = true;
        CALIBY_LOG_INFO("HNSW", "Recovery: Recovered existing index. metadata_pid=", this->metadata_pid,
                  " base_pid=", this->base_pid);
        try {
            GuardO<HNSWMetadataPage> meta_guard(this->metadata_pid);
            auto persisted_nodes = meta_guard->node_count.load(std::memory_order_acquire);
            auto persisted_level = meta_guard->max_level.load(std::memory_order_acquire);
            CALIBY_LOG_DEBUG("HNSW", "Recovery: persisted node_count=", persisted_nodes,
                      " max_level=", persisted_level,
                      " entry_point=", meta_guard->enter_point_node_id);
        } catch (const OLCRestartException&) {
            CALIBY_LOG_WARN("HNSW", "Recovery: metadata read retry failed during logging");
        }
    } else {
        if (has_existing_meta) {
            meta_info->valid = 0;
            meta_page_guard->dirty = true;
            CALIBY_LOG_INFO("HNSW", "Recovery: Existing metadata invalidated for rebuild");
        }

        const bool can_reuse_storage = has_existing_meta && meta_info->dim == Dim && meta_info->M == M &&
                                       meta_info->ef_construction == efConstruction &&
                                       meta_info->max_level == MaxLevel && meta_info->max_elements >= max_elements;
        CALIBY_LOG_DEBUG("HNSW", "Recovery: can_reuse_storage=", (can_reuse_storage ? "true" : "false"));

        if (can_reuse_storage && meta_info->metadata_pid != BufferManager::invalidPID &&
            meta_info->base_pid != BufferManager::invalidPID) {
            this->metadata_pid = meta_info->metadata_pid;
            this->base_pid = meta_info->base_pid;

            GuardX<HNSWMetadataPage> meta_guard(this->metadata_pid);
            meta_guard->max_elements = max_elements;
            meta_guard->node_count.store(0);
            meta_guard->enter_point_node_id = HNSWMetadataPage::invalid_node_id;
            meta_guard->max_level.store(0);
            meta_guard->dirty = true;
            CALIBY_LOG_INFO("HNSW", "Recovery: Reusing metadata page ", this->metadata_pid, " and base_pid ",
                      this->base_pid, "; counters reset");

            u64 total_pages = (max_elements + NodesPerPage - 1) / NodesPerPage;
            for (u64 i = 0; i < total_pages; ++i) {
                GuardX<HNSWPage> page_guard(this->base_pid + i);
                page_guard->node_count = 0;
                page_guard->dirty = true;
            }
            CALIBY_LOG_DEBUG("HNSW", "Recovery: Reset ", total_pages, " data pages");
        } else {
            meta_page_guard.release();

            AllocGuard<HNSWMetadataPage> meta_guard(allocator_);
            this->metadata_pid = meta_guard.pid;  // Already a global PID from AllocGuard

            meta_guard->dirty = false;
            meta_guard->base_pid = -1;
            meta_guard->max_elements = max_elements;
            meta_guard->node_count.store(0);
            meta_guard->alloc_count.store(0, std::memory_order_relaxed);  // Start with 0 pages allocated
            meta_guard->enter_point_node_id = HNSWMetadataPage::invalid_node_id;
            meta_guard->max_level.store(0);

            // Allocate ONLY the first page for now - pages will be allocated on-demand
            // when vectors are added. This avoids pre-allocating 10M pages for large max_elements.
            AllocGuard<HNSWPage> first_page_guard(allocator_);
            meta_guard->base_pid = first_page_guard.pid;  // Already a global PID
            first_page_guard->dirty = false;
            first_page_guard->node_count = 0;
            first_page_guard->dirty = true;
            meta_guard->alloc_count.store(1, std::memory_order_relaxed);  // 1 page allocated

            this->base_pid = meta_guard->base_pid;
            meta_guard->dirty = true;
            CALIBY_LOG_INFO("HNSW", "Recovery: Allocated new metadata page ", this->metadata_pid,
                      " base_pid=", this->base_pid, " (pages allocated on-demand)");

            // Explicitly release old guard before acquiring new one to avoid double-lock
            // Re-acquire global metadata page guard (was released at line 198)
            meta_page_guard = GuardX<MetaDataPage>(global_metadata_page_id);
            meta_page_ptr = meta_page_guard.ptr;
            meta_info = &meta_page_ptr->hnsw_meta;
        }

        meta_info->magic_value = HNSWMetaInfo::magic;
        meta_info->metadata_pid = this->metadata_pid;
        meta_info->base_pid = this->base_pid;
        meta_info->max_elements = this->max_elements_;
        meta_info->dim = Dim;
        meta_info->M = M;
        meta_info->ef_construction = efConstruction;
        meta_info->max_level = MaxLevel;
        meta_info->alloc_count.store(0, std::memory_order_relaxed);
        meta_info->valid = 1;
        meta_page_guard->dirty = true;
        CALIBY_LOG_INFO("HNSW", "Recovery: Metadata page updated and marked valid");
    }

    visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(4, this->max_elements_));
}

// --- HNSW Destructor Implementation ---
template <typename DistanceMetric>
HNSW<DistanceMetric>::~HNSW() {
    visited_list_pool_.reset(nullptr);
    search_thread_pool_.reset(nullptr);
    add_thread_pool_.reset(nullptr);
}

// --- Helper Functions (searchLayer, searchBaseLayer)---
template <typename DistanceMetric>
u32 HNSW<DistanceMetric>::getRandomLevel() {
    // Use random_device to ensure different sequences even after fork() 
    thread_local static std::mt19937 level_generator = []() {
        std::random_device rd;
        std::seed_seq seed{rd(), rd(), rd()};
        std::mt19937 gen(seed);
        return gen;
    }();
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(level_generator)) * mult_factor;
    return std::min(static_cast<u32>(r), static_cast<u32>(MaxLevel - 1));
}

template <typename DistanceMetric>
template <bool stats>
std::pair<float, u32> HNSW<DistanceMetric>::findBestEntryPointForLevel(const float* query, u32 entry_point_id, int level, float entry_point_dist) {
    u32 current_node_id = entry_point_id;
    float min_dist = entry_point_dist; // Use the pre-calculated distance
    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    std::unique_ptr<VisitedList, std::function<void(VisitedList*)>> visited_guard(
        visited_nodes, [&](VisitedList* p) { this->visited_list_pool_->releaseVisitedList(p); });

    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;

    PID pages_to_prefetch_buf[128];
    u32 offsets_within_pages_buf[128];

    bool changed = true;
    while (changed) {
        changed = false;
        u32 best_neighbor_id = current_node_id;

        try {
            GuardS<HNSWPage> current_page_guard(getNodePID(current_node_id));
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_node_id), this);

            // The first node is marked as visited here.
            // Note: We don't need to re-calculate its distance as it's passed in.
            visited_array[current_node_id] = visited_array_tag;

            if (stats) {
                stats_.search_hops.fetch_add(1, std::memory_order_relaxed);
            }
            auto neighbors_span = current_acc.getNeighbors(level, this);

            // Prefetch all neighbors' data at once before processing
            if (!neighbors_span.empty() && enable_prefetch_) {
                // for (const u32& neighbor_id : neighbors_span) {
                //     _mm_prefetch(reinterpret_cast<const char*>(&visited_array[neighbor_id]), _MM_HINT_T0);
                // }
                size_t prefetch_count = 0;
                for (const u32& neighbor_id : neighbors_span) {
                    if (!(visited_array[neighbor_id] == visited_array_tag) && prefetch_count < 128) {
                        pages_to_prefetch_buf[prefetch_count] = getNodePID(neighbor_id);
                        offsets_within_pages_buf[prefetch_count] = NodeAccessor::getVectorOffset(this, neighbor_id);
                        prefetch_count++;
                    }
                }
                if (prefetch_count > 0) {
                    bm.prefetchPages(pages_to_prefetch_buf, prefetch_count, offsets_within_pages_buf);
                }
            }

            for (int i = 0; i < (int)neighbors_span.size(); ++i) {
                const u32& neighbor_id = neighbors_span[i];

                // // In-loop prefetch for the *next* neighbor to keep the pipeline full.
                // if (i + 1 < neighbors_span.size()) {
                //     u32 next_neighbor_id = neighbors_span[i + 1];
                //     _mm_prefetch(reinterpret_cast<const char*>(&visited_array[next_neighbor_id]), _MM_HINT_T0);
                //     PID page_to_prefetch = getNodePID(next_neighbor_id);
                //     bm.prefetchPages(&page_to_prefetch, 1);
                // }

                if (!(visited_array[neighbor_id] == visited_array_tag)) {
                    try {
                        GuardORelaxed<HNSWPage> neighbor_page_guard(getNodePID(neighbor_id), index_array);
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);

                        float neighbor_dist;
                        if (stats) {
                            neighbor_dist = this->calculateDistance(query, neighbor_acc.getVector());
                        } else {
                            neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                        }

                        if (neighbor_dist < min_dist) {
                            min_dist = neighbor_dist;
                            best_neighbor_id = neighbor_id;
                            changed = true;
                        }
                    } catch (const OLCRestartException&) {
                        // CRITICAL FIX: Retry this neighbor instead of skipping it
                        i--; // Retry this neighbor if it was modified concurrently.
                        continue;
                    }
                    visited_array[neighbor_id] = visited_array_tag;
                }
            }
            current_node_id = best_neighbor_id;
        } catch (const OLCRestartException&) {
            // If a concurrent modification happens, stop the greedy search at this level
            // and proceed with the best node found so far.
            // changed = false;
        }
    }

    return {min_dist, current_node_id};
}

template <typename DistanceMetric>
template <bool stats>
std::pair<float, u32> HNSW<DistanceMetric>::searchBaseLayer(const float* query, u32 entry_point_id, int start_level, int end_level) {
    u32 current_entry_point_id = entry_point_id;
    float current_dist;
    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);
    // Calculate the distance for the initial entry point only ONCE.
    for (;;) {
        try {
            GuardORelaxed<HNSWPage> initial_guard(getNodePID(current_entry_point_id), index_array);
            NodeAccessor initial_acc(initial_guard.ptr, getNodeIndexInPage(current_entry_point_id), this);
            if (stats) {
                current_dist = this->calculateDistance(query, initial_acc.getVector());
            } else {
                current_dist = DistanceMetric::compare(query, initial_acc.getVector(), Dim);
            }
        } catch (const OLCRestartException&) {
            continue;
        }
        break;
    }
    
    for (int level = start_level; level >= end_level; --level) {
        // Pass the current best distance to the next level's search function.
        auto result = findBestEntryPointForLevel<stats>(query, current_entry_point_id, level, current_dist);
        current_dist = result.first;
        current_entry_point_id = result.second;
    }
    return {current_dist, current_entry_point_id};
}

template <typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> HNSW<DistanceMetric>::searchLayer(
    const float* query, u32 entry_point_id, u32 level, size_t ef,
    std::optional<std::pair<float, u32>> initial_entry_dist_pair) { // Added optional parameter
    
    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;
    
    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);
    
    std::vector<HeapCand> top_candidates;
    std::vector<HeapCand> candidate_queue;
    top_candidates.reserve(ef); candidate_queue.reserve(ef * 2);
    float lower_bound = std::numeric_limits<float>::max();

    // --- Use pre-calculated distance if available ---
    if (initial_entry_dist_pair && initial_entry_dist_pair->second == entry_point_id) {
        float dist = initial_entry_dist_pair->first;
        push_max_heap(top_candidates, {dist, entry_point_id});
        push_min_heap(candidate_queue, {dist, entry_point_id});
        lower_bound = dist;
        visited_array[entry_point_id] = visited_array_tag;
    } else {
        for (;;) {
            float dist;
            try {
                #ifdef HNSW_DISABLE_OPTIMISTIC_READ
                GuardS<HNSWPage> page_guard(getNodePID(entry_point_id));
                #else
                GuardORelaxed<HNSWPage> page_guard(getNodePID(entry_point_id), index_array);
                #endif
                NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(entry_point_id), this);
                if (stats) dist = this->calculateDistance(query, acc.getVector());
                else dist = DistanceMetric::compare(query, acc.getVector(), Dim);
            } catch (const OLCRestartException&) { continue; }
            push_max_heap(top_candidates, {dist, entry_point_id});
            push_min_heap(candidate_queue, {dist, entry_point_id});
            lower_bound = dist;
            visited_array[entry_point_id] = visited_array_tag;
            break;
        }
    }
    
    // The rest of the beam search logic remains the same.
    std::vector<PID> pages_to_prefetch;
    std::vector<u32> offsets_within_pages;
    pages_to_prefetch.reserve(24);
    offsets_within_pages.reserve(24);
    
    std::vector<u32> unvisited_neighbors;
    while (!candidate_queue.empty()) {
        auto [cand_dist, current_id] = pop_min_heap(candidate_queue);

        if (top_candidates.size() >= ef && cand_dist > top_candidates.front().dist) break;
        if (stats) stats_.search_hops.fetch_add(1, std::memory_order_relaxed);
        try {
            //GuardS<HNSWPage> current_page_guard(getNodePID(current_id));
            GuardO<HNSWPage> current_page_guard(getNodePID(current_id), index_array);
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_id), this);

            if (current_acc.getLevel() < level) continue;

            auto neighbors = current_acc.getNeighbors(level, this);
            // if (!neighbors.empty() && enable_prefetch_) {
            //     for (const u32& neighbor_id : neighbors) { 
            //         //PageState& next_ps = bm.getPageState(getNodePID(neighbor_id));
            //         _mm_prefetch(reinterpret_cast<const char*>(&visited_array[neighbor_id]), _MM_HINT_T2);
            //         //_mm_prefetch(reinterpret_cast<const char*>(&next_ps), _MM_HINT_T2);
            //     }
            // }
            // Prefetch all neighbors' vector data at once before processing
            // only prefetch aggressively for lower levels as the top levels have exponentially fewer nodes and are likely to be cached
            unvisited_neighbors.clear();

            if (level <= 2 && !neighbors.empty() && enable_prefetch_) {
                for (const u32& neighbor_id : neighbors) { 
                    _mm_prefetch(reinterpret_cast<const char*>(&visited_array[neighbor_id]), _MM_HINT_T0);
                }
                pages_to_prefetch.clear();
                offsets_within_pages.clear();
                // unroll the following loop for better performance
                size_t i = 0;
                for (i = 0; i + 4 < neighbors.size();) {
                    const u32& neighbor_id1 = neighbors[i];
                    if (!(visited_array[neighbor_id1] == visited_array_tag)) {
                        pages_to_prefetch.push_back(getNodePID(neighbor_id1));
                        offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id1));
                        unvisited_neighbors.push_back(neighbor_id1);
                    }
                    const u32& neighbor_id2 = neighbors[i + 1];
                    if (!(visited_array[neighbor_id2] == visited_array_tag)) {
                        pages_to_prefetch.push_back(getNodePID(neighbor_id2));
                        offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id2));
                        unvisited_neighbors.push_back(neighbor_id2);
                    }
                    const u32& neighbor_id3 = neighbors[i + 2];
                    if (!(visited_array[neighbor_id3] == visited_array_tag)) {
                        pages_to_prefetch.push_back(getNodePID(neighbor_id3));
                        offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id3));
                        unvisited_neighbors.push_back(neighbor_id3);
                    }
                    const u32& neighbor_id4 = neighbors[i + 3];
                    if (!(visited_array[neighbor_id4] == visited_array_tag)) {
                        pages_to_prefetch.push_back(getNodePID(neighbor_id4));
                        offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id4));
                        unvisited_neighbors.push_back(neighbor_id4);
                    }
                    if (i + 4 >= neighbors.size()) {
                        break;
                    }
                    i += 4;
                }
                for (; i < neighbors.size(); ++i) {
                    const u32& neighbor_id = neighbors[i];
                    if (!(visited_array[neighbor_id] == visited_array_tag)) {
                        pages_to_prefetch.push_back(getNodePID(neighbor_id));
                        offsets_within_pages.push_back(NodeAccessor::getVectorOffset(this, neighbor_id));
                        unvisited_neighbors.push_back(neighbor_id);
                    }
                }

                bm.prefetchPages(pages_to_prefetch.data(), pages_to_prefetch.size(), offsets_within_pages.data());
            } else {
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    const u32& neighbor_id = neighbors[i];
                    if (!(visited_array[neighbor_id] == visited_array_tag)) {
                        unvisited_neighbors.push_back(neighbor_id);
                    }
                }
            }

            std::span<const u32> unvisited_neighbor_span(unvisited_neighbors.data(), unvisited_neighbors.size());
            // u32 next_neighbor_id = neighbors[0];
            // _mm_prefetch(reinterpret_cast<const char*>(&visited_array[next_neighbor_id]), _MM_HINT_T0);
            // PID page_to_prefetch = getNodePID(next_neighbor_id);
            // u32 next_neighbor_off = NodeAccessor::getVectorOffset(this, next_neighbor_id);
            // bm.prefetchPages(&page_to_prefetch, 1, &next_neighbor_off);
            for (int i = 0; i < (int)unvisited_neighbor_span.size(); ++i) {
                const u32& neighbor_id = unvisited_neighbor_span[i];
                // if (i + 1 < unvisited_neighbor_span.size() && enable_prefetch_) {
                //     u32 next_neighbor_id = unvisited_neighbor_span[i + 1];
                //     //_mm_prefetch(reinterpret_cast<const char*>(&visited_array[next_neighbor_id]), _MM_HINT_T0);
                //     PID page_to_prefetch = getNodePID(next_neighbor_id);
                //     u32 next_neighbor_off = NodeAccessor::getVectorOffset(this, next_neighbor_id);
                //     char* pg = (char*)bm.toPtr(page_to_prefetch); // ensure the page is resident
                //     //mm_prefetch
                //     _mm_prefetch(reinterpret_cast<const char*>(pg + next_neighbor_off), _MM_HINT_T0);
                // }

                // Skip if already visited (can happen on OLC retry)
                if (visited_array[neighbor_id] == visited_array_tag) {
                    continue;
                }
                
                float neighbor_dist;
                try{
                    #ifdef HNSW_DISABLE_OPTIMISTIC_READ
                    GuardS<HNSWPage> neighbor_page_guard(getNodePID(neighbor_id));
                    #else
                    // Use specialized constructor with IndexTranslationArray to avoid TLS cache lookups
                    PID neighbor_pid = getNodePID(neighbor_id);
                    // GuardO<HNSWPage> neighbor_page_guard = (index_array != nullptr) 
                    //     ? GuardO<HNSWPage>(neighbor_pid, index_array) 
                    //     : GuardO<HNSWPage>(neighbor_pid);
                    GuardORelaxed<HNSWPage> neighbor_page_guard(neighbor_pid, index_array);
                    #endif
                    NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                    // if (stats) {
                    //     neighbor_dist = this->calculateDistance(query, neighbor_acc.getVector());
                    // } else {
                        neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                    // }
                } catch(const OLCRestartException&){
                    // CRITICAL FIX: Retry this neighbor instead of skipping it
                    // Skipping causes nodes to become unreachable during concurrent modifications
                    i--; // Retry this neighbor if it was modified concurrently.
                    continue;
                }

                if (neighbor_dist < lower_bound || top_candidates.size() < ef) {
                    push_min_heap(candidate_queue, {neighbor_dist, neighbor_id});
                    if (top_candidates.size() < ef) {
                        push_max_heap(top_candidates, {neighbor_dist, neighbor_id});
                    } else if (neighbor_dist < top_candidates.front().dist) {
                        replace_top_max_heap(top_candidates, {neighbor_dist, neighbor_id});
                    }
                    if (top_candidates.size() >= ef) lower_bound = top_candidates.front().dist;
                }
                visited_array[neighbor_id] = visited_array_tag;
            }
        } catch (const OLCRestartException&) {
            push_min_heap(candidate_queue, {cand_dist, current_id});
            continue; // Skip this node if it was modified concurrently.
            //std::cout << "OLCRestartException caught in searchLayer" << std::endl;
        }
    }

    std::vector<std::pair<float, u32>> results;
    results.reserve(top_candidates.size());
    std::sort(top_candidates.begin(), top_candidates.end(),
              [](const HeapCand& a, const HeapCand& b) { return a.dist < b.dist; });
    for (auto& hc : top_candidates) results.emplace_back(hc.dist, hc.id);
    visited_list_pool_->releaseVisitedList(visited_nodes);
    return results;
}

template <typename DistanceMetric>
inline PID HNSW<DistanceMetric>::getNodePID(u32 node_id) const {
    return base_pid + (node_id / NodesPerPage);
}

template <typename DistanceMetric>
inline u32 HNSW<DistanceMetric>::getNodeIndexInPage(u32 node_id) const {
    return node_id % NodesPerPage;
}

template <typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<DistanceMetric>::selectNeighborsHeuristic(
    const float* query, const std::vector<std::pair<float, u32>>& candidates, size_t M_limit) {
    if (candidates.size() <= M_limit) {
        return candidates;
    }
    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

    std::vector<std::pair<float, u32>> result;
    result.reserve(M_limit);

    // candidates vector is already sorted by distance, from closest to farthest.
    // We can iterate through it directly.
    for (const auto& current_candidate : candidates) {
        if (result.size() >= M_limit) {
            break;
        }
        bool is_good_candidate = true;
        while (true) {
            // Check against the neighbors we have already selected for the result list.

            try {
                for (const auto& selected : result) {
                    GuardORelaxed<HNSWPage> candidate_page_guard(getNodePID(current_candidate.second), index_array);
                    NodeAccessor candidate_acc(candidate_page_guard.ptr, getNodeIndexInPage(current_candidate.second),
                                            this);
                    const float* candidate_vector = candidate_acc.getVector();

                    GuardORelaxed<HNSWPage> selected_page_guard(getNodePID(selected.second), index_array);
                    NodeAccessor selected_acc(selected_page_guard.ptr, getNodeIndexInPage(selected.second), this);
                    const float* selected_vector = selected_acc.getVector();

                    if (DistanceMetric::compare(candidate_vector, selected_vector, Dim) < current_candidate.first) {
                        is_good_candidate = false;
                        break;
                    }
                }
                break;
            } catch (const OLCRestartException&) {
                is_good_candidate = true;
                continue;
            }
        }
        

        if (is_good_candidate) {
            result.push_back(current_candidate);
        }
    }

    return result;
}

template <typename DistanceMetric>
void HNSW<DistanceMetric>::addPoint_internal(const float* point, u32 new_node_id) {
    const u32 new_node_level = getRandomLevel();
    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

    // --- 0. Ensure the page for this node exists (on-demand allocation) ---
    u64 required_page_num = new_node_id / NodesPerPage;
    {
        // Check if we need to allocate more pages
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        u64 current_alloc_count = meta_guard->alloc_count.load(std::memory_order_acquire);
        
        if (required_page_num >= current_alloc_count) {
            // Need to initialize pages from current_alloc_count up to required_page_num (inclusive)
            // We DON'T use AllocGuard here because after recovery, the allocator counter
            // may be at a different position. Instead, we directly access pages by their
            // computed PID (base_pid + page_idx). The buffer manager will create the page
            // if it doesn't exist (readPage handles missing pages by zeroing them).
            for (u64 page_idx = current_alloc_count; page_idx <= required_page_num; ++page_idx) {
                PID page_pid = base_pid + page_idx;
                
                // Access the page directly - this will create it if it doesn't exist
                GuardX<HNSWPage> new_page_guard(page_pid);
                
                // Initialize the new page
                new_page_guard->node_count = 0;
                new_page_guard->dirty = true;
            }
            
            // Update alloc_count to include all newly allocated pages
            u64 new_alloc_count = required_page_num + 1;
            meta_guard->alloc_count.store(new_alloc_count, std::memory_order_release);
            meta_guard->dirty = true;
            
            // CRITICAL: Also update IndexTranslationArray::allocCount so flushAll knows
            // about these pages. Without this, pages won't be flushed on close!
            // The base_pid encodes index_id in high bits and local PID 2 in low bits.
            // We need to update the allocCount to be: (number of metadata pages) + (number of data pages)
            // base_pid local part + new_alloc_count gives us the next local PID to use
            u64 base_local_pid = base_pid & 0xFFFFFFFF;  // Extract local PID from base_pid
            u64 total_local_pages = base_local_pid + new_alloc_count;  // Total pages used by this index
            if (index_array) {
                // Atomically update allocCount to the maximum of current and required
                u64 current_index_alloc = index_array->allocCount.load(std::memory_order_acquire);
                while (current_index_alloc < total_local_pages) {
                    if (index_array->allocCount.compare_exchange_weak(current_index_alloc, total_local_pages,
                            std::memory_order_release, std::memory_order_acquire)) {
                        break;
                    }
                }
            }
        }
    }

    PID new_node_pid = getNodePID(new_node_id);

    // --- 1. Allocate space and initialize data for the new node on its page ---
    {
        GuardX<HNSWPage> page_guard(new_node_pid);
        HNSWPage* page = page_guard.ptr;
        u32 node_idx = getNodeIndexInPage(new_node_id);

        if (node_idx >= MaxNodesPerPage) {
            throw std::runtime_error("Node index exceeds page capacity. This should not happen with pre-calculation.");
        }

        // Calculate direct offset to this node's data
        u8* node_start = page->getNodeData() + node_idx * FixedNodeSize;

        // Initialize node: [Vector][Level Counts][Neighbor IDs]
        float* vector_ptr = reinterpret_cast<float*>(node_start);
        u16* level_counts_ptr = reinterpret_cast<u16*>(node_start + VectorSize);
        
        // Copy vector data
        memcpy(vector_ptr, point, VectorSize);

        // Zero-initialize level counts and neighbor IDs
        memset(level_counts_ptr, 0, MaxNeighborsHeaderSize + MaxNeighborsListSize);

        // Update page metadata
        page->node_count = std::max(page->node_count, static_cast<u16>(node_idx + 1));
        page->dirty = true;
    }

    // --- 2. Find the global entry point for the search ---
    u32 enter_point_id;
    u32 max_l;
    
    // Check if this might be the first node (early check without lock)
    bool might_be_first = false;
    for (;;) {  // OLC retry loop for reading metadata
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            might_be_first = (enter_point_id == HNSWMetadataPage::invalid_node_id);
        } catch (const OLCRestartException&) {
            continue;
        }
        break;
    }

    // If this is the first node, set it as the entry point and return.
    // CRITICAL: Must re-check inside exclusive lock to prevent race conditions!
    if (might_be_first) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        if (meta_guard->enter_point_node_id == HNSWMetadataPage::invalid_node_id) {
            // Still invalid, we are the first node
            meta_guard->enter_point_node_id = new_node_id;
            meta_guard->max_level.store(new_node_level, std::memory_order_release);
            meta_guard->dirty = true;
            return;
        } else {
            // Another thread beat us to it, re-read the entry point
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
        }
    }

    // --- 3. Search from top layers down to find the best entry point for the new node's level ---
    std::optional<std::pair<float, u32>> entry_point_with_dist_opt = std::nullopt;
    if (new_node_level < max_l) {
        auto result = searchBaseLayer(point, enter_point_id, max_l, new_node_level + 1);
        enter_point_id = result.second;
        entry_point_with_dist_opt = result; // Store the pair for the first searchLayer call.
    } else {
        entry_point_with_dist_opt = std::nullopt;
    }
    std::vector<float> neighbor_vector_copy(Dim);
    std::vector<u32> current_neighbors_ids; // Using vector for potential modification.
    std::unordered_map<u32, std::vector<float>> connection_vectors;
    // --- 4. Connection phase: from new_node_level down to 0 ---
    for (int level = std::min(static_cast<u32>(new_node_level), max_l); level >= 0; --level) {
        // Find the best neighbors for the new node at the current level.
        // Pass the pre-calculated distance on the FIRST iteration.
        auto candidates = searchLayer(point, enter_point_id, level, efConstruction, entry_point_with_dist_opt);
        
        // The pre-calculated distance is only valid for the first iteration.
        // Clear it for all subsequent iterations of this loop.
        if (entry_point_with_dist_opt.has_value()) {
            entry_point_with_dist_opt = std::nullopt;
        }

        if (candidates.empty()) {
            continue;
        }

        size_t M_level = (level == 0) ? M0 : M;
        auto neighbors_to_link_pairs = selectNeighborsHeuristic(point, candidates, M_level);

        // *** FINE-GRAINED LOCKING ***
        // Step 4a: Link the new node TO its chosen neighbors.
        {
            GuardX<HNSWPage> new_node_page_guard(new_node_pid);
            MutableNodeAccessor new_node_acc(new_node_page_guard.ptr, getNodeIndexInPage(new_node_id), this);
            std::vector<u32> neighbor_ids;
            neighbor_ids.reserve(neighbors_to_link_pairs.size());
            for (const auto& p : neighbors_to_link_pairs) {
                neighbor_ids.push_back(p.second);
            }
            new_node_acc.setNeighbors(level, neighbor_ids, this);
        }

        // Step 4b: Link neighbors BACK to the new node, one by one.
        for (const std::pair<float, u32>& neighbor_pair : neighbors_to_link_pairs) {
            const float dist_point_to_neighbor = neighbor_pair.first;
            const u32 neighbor_id = neighbor_pair.second;
            PID neighbor_pid = getNodePID(neighbor_id);

            for (;;) {  // OLC retry loop for updating a single neighbor
                try {
                    // --- PHASE 1: READ EVERYTHING NEEDED, WITHOUT ANY EXCLUSIVE LOCKS ---
                    current_neighbors_ids.clear();
                    connection_vectors.clear();
                    bool needs_pruning;
                    
                    {
                        GuardO<HNSWPage> neighbor_page_guard(neighbor_pid, index_array);
                        NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        
                        const float* neighbor_vec_ptr = neighbor_acc.getVector();
                        std::copy(neighbor_vec_ptr, neighbor_vec_ptr + Dim, neighbor_vector_copy.begin());

                        auto neighbors_span = neighbor_acc.getNeighbors(level, this);
                        current_neighbors_ids.assign(neighbors_span.begin(), neighbors_span.end());
                    }

                    needs_pruning = (current_neighbors_ids.size() >= M_level);

                    if (needs_pruning) {
                        // for (u32 conn_id : current_neighbors_ids) {
                        //     GuardORelaxed<HNSWPage> conn_page_guard(getNodePID(conn_id), index_array);
                        //     NodeAccessor conn_acc(conn_page_guard.ptr, getNodeIndexInPage(conn_id), this);
                        //     const float* conn_vec_ptr = conn_acc.getVector();
                        //     connection_vectors[conn_id].assign(conn_vec_ptr, conn_vec_ptr + Dim);
                        // }
                    }

                    // --- PHASE 2: ACQUIRE A SINGLE LOCK AND WRITE ---
                    {
                        GuardX<HNSWPage> neighbor_page_guard(neighbor_pid);
                        NodeAccessor locked_reader(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                        auto latest_neighbors_span = locked_reader.getNeighbors(level, this);

                        if (latest_neighbors_span.size() >= M_level) {
                            // --- Pruning Path ---
                            std::priority_queue<std::pair<float, u32>> connections_to_prune;
                            
                            connections_to_prune.push({dist_point_to_neighbor, new_node_id});
                            
                            for (u32 conn_id : latest_neighbors_span) {
                                // auto it = connection_vectors.find(conn_id);
                                // if (it != connection_vectors.end()) {
                                //     const std::vector<float>& conn_vector = it->second;
                                //     float dist = DistanceMetric::compare(neighbor_vector_copy.data(), conn_vector.data(), Dim);
                                //     connections_to_prune.push({dist, conn_id});
                                // }
                                GuardORelaxed<HNSWPage> conn_page_guard(getNodePID(conn_id), index_array);
                                NodeAccessor conn_acc(conn_page_guard.ptr, getNodeIndexInPage(conn_id), this);
                                const float* conn_vec_ptr = conn_acc.getVector();
                                float dist = DistanceMetric::compare(neighbor_vector_copy.data(), conn_vec_ptr, Dim);
                                connections_to_prune.push({dist, conn_id});
                            }

                            while (connections_to_prune.size() > M_level) {
                                connections_to_prune.pop();
                            }

                            std::vector<u32> new_neighbor_list;
                            new_neighbor_list.reserve(M_level);
                            while (!connections_to_prune.empty()) {
                                new_neighbor_list.push_back(connections_to_prune.top().second);
                                connections_to_prune.pop();
                            }

                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                            neighbor_acc.setNeighbors(level, new_neighbor_list, this);
                        } else {
                            // --- Non-Pruning Path ---
                            MutableNodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                            neighbor_acc.addNeighbor(level, new_node_id, this);
                        }
                    }
                    break; // Success, exit retry loop
                } catch (const OLCRestartException&) {
                    continue; // An optimistic lock failed, restart the process for this neighbor.
                }
            } // End of retry loop
        } // End of for-each neighbor loop

        // Update the entry point for the next level down
        if (!neighbors_to_link_pairs.empty()) {
            enter_point_id = neighbors_to_link_pairs.front().second;
        }
    }

    // --- CONNECTIVITY FIX: Force bidirectional connectivity to anchor (node 0) ---
    // This ensures all nodes can reach node 0 and early warmup nodes remain reachable.
    // We use FORCE addition which replaces the furthest neighbor if the list is full.
    {
        const u32 ANCHOR_NODE = 0;
        if (new_node_id != ANCHOR_NODE) {
            // FORCE link: new_node -> anchor at level 0
            PID new_node_pid = getNodePID(new_node_id);
            for (;;) {
                try {
                    GuardX<HNSWPage> new_guard(new_node_pid);
                    MutableNodeAccessor new_acc(new_guard.ptr, getNodeIndexInPage(new_node_id), this);
                    
                    // Check if already linked to anchor
                    std::span<const u32> new_neighbors = new_acc.getNeighbors(0, this);
                    bool has_anchor = false;
                    for (u32 n : new_neighbors) {
                        if (n == ANCHOR_NODE) {
                            has_anchor = true;
                            break;
                        }
                    }
                    
                    if (!has_anchor) {
                        if (!new_acc.addNeighbor(0, ANCHOR_NODE, this)) {
                            // List is full - force add by replacing furthest non-anchor neighbor
                            const float* new_vector = new_acc.getVector();
                            float max_dist = -1.0f;
                            u32 furthest = HNSWMetadataPage::invalid_node_id;
                            
                            for (u32 neighbor_id : new_neighbors) {
                                if (neighbor_id == ANCHOR_NODE) continue; // Don't remove anchor
                                PID neigh_pid = getNodePID(neighbor_id);
                                GuardORelaxed<HNSWPage> neigh_guard(neigh_pid, index_array);
                                NodeAccessor neigh_acc(neigh_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                                float dist = DistanceMetric::compare(new_vector, neigh_acc.getVector(), Dim);
                                if (dist > max_dist) {
                                    max_dist = dist;
                                    furthest = neighbor_id;
                                }
                            }
                            
                            if (furthest != HNSWMetadataPage::invalid_node_id) {
                                // Replace furthest with anchor
                                std::vector<u32> updated_neighbors;
                                updated_neighbors.reserve(M0);
                                updated_neighbors.push_back(ANCHOR_NODE);
                                for (u32 n : new_neighbors) {
                                    if (n != furthest) {
                                        updated_neighbors.push_back(n);
                                    }
                                }
                                new_acc.setNeighbors(0, updated_neighbors, this);
                            }
                        }
                    }
                    break;
                } catch (const OLCRestartException&) {
                    continue;
                }
            }
        }
    }

    // --- 5. Update global entry point if the new node is the highest ---
    if (new_node_level > max_l) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        u32 old_entry_point = meta_guard->enter_point_node_id;
        if (new_node_level > meta_guard->max_level.load(std::memory_order_acquire)) {
            meta_guard->enter_point_node_id = new_node_id;
            meta_guard->max_level.store(new_node_level, std::memory_order_release);
            meta_guard->dirty = true;
            
            // CRITICAL: Ensure old entry point remains reachable from new entry point
            // When entry point changes, we must create a bidirectional link at level 0
            // to prevent the old subgraph from becoming orphaned
            if (old_entry_point != HNSWMetadataPage::invalid_node_id && old_entry_point != new_node_id) {
                // Link new_node -> old_entry_point at level 0
                {
                    PID new_node_pid = getNodePID(new_node_id);
                    GuardX<HNSWPage> new_guard(new_node_pid);
                    MutableNodeAccessor new_acc(new_guard.ptr, getNodeIndexInPage(new_node_id), this);
                    new_acc.addNeighbor(0, old_entry_point, this); // Ignore if full, bidirectional link below is more important
                }
                // Link old_entry_point -> new_node at level 0 (CRITICAL: ensures reachability)
                {
                    PID old_pid = getNodePID(old_entry_point);
                    GuardX<HNSWPage> old_guard(old_pid);
                    MutableNodeAccessor old_acc(old_guard.ptr, getNodeIndexInPage(old_entry_point), this);
                    if (!old_acc.addNeighbor(0, new_node_id, this)) {
                        // Old entry point's list is full - replace furthest neighbor
                        const float* old_vector = old_acc.getVector();
                        std::span<const u32> current_neighbors = old_acc.getNeighbors(0, this);
                        
                        // Find furthest neighbor that is NOT node 0 (anchor) or the new node
                        float max_dist = -1;
                        u32 furthest = HNSWMetadataPage::invalid_node_id;
                        for (u32 neighbor_id : current_neighbors) {
                            if (neighbor_id == 0 || neighbor_id == new_node_id) continue; // Protect anchor and new node
                            PID neigh_pid = getNodePID(neighbor_id);

                            GuardORelaxed<HNSWPage> neigh_guard(neigh_pid, index_array);
                            NodeAccessor neigh_acc(neigh_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                            float dist = DistanceMetric::compare(old_vector, neigh_acc.getVector(), Dim);
                            if (dist > max_dist) {
                                max_dist = dist;
                                furthest = neighbor_id;
                            }
                        }
                        
                        if (furthest != HNSWMetadataPage::invalid_node_id) {
                            // Replace furthest with new entry point
                            std::vector<u32> new_neighbors;
                            new_neighbors.reserve(M0);
                            new_neighbors.push_back(new_node_id); // Add new entry point first
                            for (u32 neighbor_id : current_neighbors) {
                                if (neighbor_id != furthest) {
                                    new_neighbors.push_back(neighbor_id);
                                }
                            }
                            old_acc.setNeighbors(0, new_neighbors, this);
                        }
                    }
                }
                
                // CRITICAL: Also ensure new entry point has node 0 (anchor) as neighbor
                // This guarantees node 0 is reachable in 1 hop from entry point
                {
                    PID new_node_pid = getNodePID(new_node_id);
                    GuardX<HNSWPage> new_guard(new_node_pid);
                    MutableNodeAccessor new_acc(new_guard.ptr, getNodeIndexInPage(new_node_id), this);
                    
                    // Check if already linked to node 0
                    std::span<const u32> neighbors = new_acc.getNeighbors(0, this);
                    bool has_anchor = false;
                    for (u32 n : neighbors) {
                        if (n == 0) {
                            has_anchor = true;
                            break;
                        }
                    }
                    
                    if (!has_anchor) {
                        if (!new_acc.addNeighbor(0, 0, this)) {
                            // List is full - force add by replacing furthest neighbor (except node 0)
                            const float* new_vector = new_acc.getVector();
                            float max_dist = -1.0f;
                            u32 furthest = HNSWMetadataPage::invalid_node_id;
                            
                            for (u32 neighbor_id : neighbors) {
                                if (neighbor_id == 0) continue;
                                PID neigh_pid = getNodePID(neighbor_id);
                                GuardORelaxed<HNSWPage> neigh_guard(neigh_pid, index_array);
                                NodeAccessor neigh_acc(neigh_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                                float dist = DistanceMetric::compare(new_vector, neigh_acc.getVector(), Dim);
                                if (dist > max_dist) {
                                    max_dist = dist;
                                    furthest = neighbor_id;
                                }
                            }
                            
                            if (furthest != HNSWMetadataPage::invalid_node_id) {
                                std::vector<u32> updated_neighbors;
                                updated_neighbors.reserve(M0);
                                updated_neighbors.push_back(0); // Anchor first
                                for (u32 n : neighbors) {
                                    if (n != furthest) {
                                        updated_neighbors.push_back(n);
                                    }
                                }
                                new_acc.setNeighbors(0, updated_neighbors, this);
                            }
                        }
                    }
                }
            }
        }
    }
}

// --- Parallel and Single Add/Search Functions (unchanged) ---
template <typename DistanceMetric>
void HNSW<DistanceMetric>::addPoint_parallel(std::span<const float> points, size_t num_threads) {
    if (points.empty()) return;
    if (points.size() % Dim != 0) {
        throw std::invalid_argument("Total number of floats in points span is not a multiple of the vector dimension.");
    }
    size_t num_points = points.size() / Dim;
    u32 start_id;
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        start_id = meta_guard->node_count.load();
        if (start_id + num_points > max_elements_) {
            throw std::runtime_error("Cannot add items; index would exceed max_elements.");
        }
        meta_guard->node_count.store(start_id + num_points);
        meta_guard->dirty = true;
    }
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_points);
    if (threads_to_use <= 1) {
        for (size_t i = 0; i < num_points; ++i) {
            addPoint_internal(points.data() + i * Dim, start_id + i);
        }
    } else {
        // Insert a significant portion of nodes single-threaded to establish robust graph structure
        // 10% warmup ensures early nodes have stable, well-connected neighborhoods that won't be
        // disrupted by later parallel insertions. This is needed because back-link pruning during
        // parallel insertion can cause early nodes to lose incoming edges.
        size_t warmup_size = 0;
        for (size_t i = 0; i < warmup_size; ++i) {
            addPoint_internal(points.data() + i * Dim, start_id + i);
        }
        
        // Now insert the remaining points in parallel
        size_t remaining_points = num_points - warmup_size;
        if (remaining_points > 0) {
            // Reuse thread pool
            ThreadPool* pool = getOrCreateAddPool(threads_to_use);
            std::vector<std::future<void>> futures;
            futures.reserve(threads_to_use);
            
            // Pre-partition work: each thread gets a contiguous range of points.
            size_t chunk = remaining_points / threads_to_use;
            size_t remainder = remaining_points % threads_to_use;

            for (size_t thread_idx = 0; thread_idx < threads_to_use; ++thread_idx) {
                size_t start = thread_idx * chunk + std::min(thread_idx, remainder);
                size_t extra = (thread_idx < remainder) ? 1 : 0;
                size_t end = start + chunk + extra; // [start, end)

                futures.emplace_back(pool->enqueue([this, points_data = points.data(), start_id, warmup_size, start, end]() {
                for (size_t point_idx = start; point_idx < end; ++point_idx) {
                    size_t actual_idx = warmup_size + point_idx;  // Offset by warmup size
                    const float* point_vector = points_data + actual_idx * Dim;
                    u32 node_id = start_id + actual_idx;
                    this->addPoint_internal(point_vector, node_id);
                }
                }));
            }
            
            for (auto& future : futures) {
                future.get();
            }
        }
    }
}

template <typename DistanceMetric>
void HNSW<DistanceMetric>::addPointsWithIdsParallel(const std::vector<const float*>& data_ptrs,
                                                     const std::vector<uint32_t>& ids,
                                                     size_t num_threads) {
    if (data_ptrs.empty() || ids.empty()) return;
    if (data_ptrs.size() != ids.size()) {
        throw std::invalid_argument("data_ptrs and ids must have the same size.");
    }
    
    size_t num_points = data_ptrs.size();
    
    // Update node count to max(current, max_id + 1) to ensure all IDs are valid
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        u32 max_id = *std::max_element(ids.begin(), ids.end());
        u32 current_count = meta_guard->node_count.load();
        if (max_id >= current_count) {
            if (max_id >= max_elements_) {
                throw std::runtime_error("Cannot add items; node_id would exceed max_elements.");
            }
            meta_guard->node_count.store(max_id + 1);
        }
        meta_guard->dirty = true;
    }
    
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_points);
    
    if (threads_to_use <= 1) {
        // Single-threaded insertion
        for (size_t i = 0; i < num_points; ++i) {
            addPoint_internal(data_ptrs[i], ids[i]);
        }
    } else {
        // Parallel insertion using thread pool
        ThreadPool* pool = getOrCreateAddPool(threads_to_use);
        std::vector<std::future<void>> futures;
        futures.reserve(threads_to_use);
        
        // Pre-partition work: each thread gets a contiguous range of points
        size_t chunk = num_points / threads_to_use;
        size_t remainder = num_points % threads_to_use;
        
        for (size_t thread_idx = 0; thread_idx < threads_to_use; ++thread_idx) {
            size_t start = thread_idx * chunk + std::min(thread_idx, remainder);
            size_t extra = (thread_idx < remainder) ? 1 : 0;
            size_t end = start + chunk + extra;
            
            futures.emplace_back(pool->enqueue([this, &data_ptrs, &ids, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    this->addPoint_internal(data_ptrs[i], ids[i]);
                }
            }));
        }
        
        for (auto& future : futures) {
            future.get();
        }
    }
}

template <typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<DistanceMetric>::computeDistancesToCandidates(
    const float* query, const std::vector<uint64_t>& candidate_ids, size_t k) {
    
    // Use a max-heap to keep track of top-k smallest distances
    std::priority_queue<std::pair<float, u32>> top_k_heap;
    
    for (uint64_t candidate_id : candidate_ids) {
        u32 node_id = static_cast<u32>(candidate_id);
        
        if (node_id >= max_elements_) {
            continue;
        }
        
        PID node_pid = getNodePID(node_id);
        
        try {
            GuardO<HNSWPage> page_guard(node_pid);
            NodeAccessor node_acc(page_guard.ptr, getNodeIndexInPage(node_id), this);
            const float* node_vector = node_acc.getVector();
            float dist = calculateDistance(query, node_vector);
            
            if (top_k_heap.size() < k) {
                top_k_heap.emplace(dist, node_id);
            } else if (dist < top_k_heap.top().first) {
                top_k_heap.pop();
                top_k_heap.emplace(dist, node_id);
            }
        } catch (...) {
            continue;
        }
    }
    
    std::vector<std::pair<float, u32>> results;
    results.reserve(top_k_heap.size());
    while (!top_k_heap.empty()) {
        results.push_back(top_k_heap.top());
        top_k_heap.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

/**
 * Filtered HNSW search: Traverse the HNSW graph normally but only accept 
 * nodes that pass the filter predicate into the result set.
 * 
 * This is much faster than brute-force when selectivity is moderate (e.g. 1-50%)
 * because it only computes distances to O(ef_search * expansion) nodes rather
 * than all matching candidates.
 * 
 * The filter_fn returns true if the node_id passes the filter.
 * ef_search is expanded proportionally to (1/selectivity) to compensate for
 * filtered-out nodes.
 */
template <typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<DistanceMetric>::searchKnnFiltered(
    const float* query, size_t k, size_t ef_search_param,
    const std::function<bool(u32)>& filter_fn) {
    
    // --- 1. Get the global entry point ---
    u32 enter_point_id;
    u32 max_l;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        return {};
    }

    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

    // --- 2. Calculate initial distance to entry point ---
    float entry_dist;
    for (;;) {
        try {
            GuardORelaxed<HNSWPage> initial_guard(getNodePID(enter_point_id), index_array);
            NodeAccessor initial_acc(initial_guard.ptr, getNodeIndexInPage(enter_point_id), this);
            entry_dist = DistanceMetric::compare(query, initial_acc.getVector(), Dim);
        } catch (const OLCRestartException&) {
            continue;
        }
        break;
    }

    // --- 3. Greedily search upper layers (no filtering here) ---
    std::pair<float, u32> entry_point_for_layer0 = {entry_dist, enter_point_id};
    
    if (max_l > 0) {
        entry_point_for_layer0 = searchBaseLayer(query, enter_point_id, max_l, 1);
    }
    
    // --- 4. Beam search on layer 0 with inline filtering ---
    size_t ef_search = (ef_search_param == 0) ? std::max(static_cast<size_t>(efConstruction), k) : ef_search_param;
    ef_search = std::max(ef_search, k);
    
    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;
    
    // top_candidates: all good candidates found (unfiltered, for graph traversal)
    // filtered_results: only candidates that pass the filter (for final result)
    std::priority_queue<std::pair<float, u32>> top_candidates;
    std::priority_queue<std::pair<float, u32>> filtered_results; // max-heap, top-k filtered
    std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, 
                        std::greater<std::pair<float, u32>>> candidate_queue;
    
    float entry_d = entry_point_for_layer0.first;
    u32 entry_id = entry_point_for_layer0.second;
    
    top_candidates.push({entry_d, entry_id});
    candidate_queue.push({entry_d, entry_id});
    visited_array[entry_id] = visited_array_tag;
    
    // Check if entry point passes filter
    if (filter_fn(entry_id)) {
        filtered_results.push({entry_d, entry_id});
    }
    
    while (!candidate_queue.empty()) {
        auto current_pair = candidate_queue.top();
        candidate_queue.pop();

        // Stop if current candidate is worse than worst in top_candidates
        if (top_candidates.size() >= ef_search && current_pair.first > top_candidates.top().first) {
            break;
        }
        
        u32 current_id = current_pair.second;
        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_id), index_array);
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_id), this);

            if (current_acc.getLevel() < 0) continue;

            auto neighbors = current_acc.getNeighbors(0, this);
            
            for (size_t i = 0; i < neighbors.size(); ++i) {
                const u32 neighbor_id = neighbors[i];
                
                if (visited_array[neighbor_id] == visited_array_tag) {
                    continue;
                }
                visited_array[neighbor_id] = visited_array_tag;
                
                float neighbor_dist;
                try {
                    PID neighbor_pid = getNodePID(neighbor_id);
                    GuardORelaxed<HNSWPage> neighbor_page_guard(neighbor_pid, index_array);
                    NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                    neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                } catch (const OLCRestartException&) {
                    i--; // Retry
                    continue;
                }

                // Always add to graph traversal structures (unfiltered)
                if (top_candidates.size() < ef_search || neighbor_dist < top_candidates.top().first) {
                    candidate_queue.push({neighbor_dist, neighbor_id});
                    top_candidates.push({neighbor_dist, neighbor_id});
                    if (top_candidates.size() > ef_search) {
                        top_candidates.pop();
                    }
                }
                
                // Add to filtered results only if passes filter
                if (filter_fn(neighbor_id)) {
                    if (filtered_results.size() < k) {
                        filtered_results.push({neighbor_dist, neighbor_id});
                    } else if (neighbor_dist < filtered_results.top().first) {
                        filtered_results.pop();
                        filtered_results.push({neighbor_dist, neighbor_id});
                    }
                }
            }
        } catch (const OLCRestartException&) {
            candidate_queue.push(current_pair);
            continue;
        }
    }

    visited_list_pool_->releaseVisitedList(visited_nodes);

    // Extract filtered results sorted by distance ascending
    std::vector<std::pair<float, u32>> results;
    results.reserve(filtered_results.size());
    while (!filtered_results.empty()) {
        results.push_back(filtered_results.top());
        filtered_results.pop();
    }
    std::reverse(results.begin(), results.end());
    
    return results;
}

/**
 * ACORN-inspired filtered HNSW search with:
 * 1. 2-hop neighbor expansion to traverse predicate subgraph
 * 2. Multiple random entry points from matching set
 * 3. Dynamic ef_search scaling based on selectivity
 * 
 * Based on the ACORN paper: "ACORN: Performant and Predicate-Agnostic Search Over 
 * Vector Embeddings and Structured Data" (Patel et al., SIGMOD 2024)
 */
template <typename DistanceMetric>
std::vector<std::pair<float, u32>> HNSW<DistanceMetric>::searchKnnFilteredACORN(
    const float* query, size_t k, size_t ef_search_param,
    const std::function<bool(u32)>& filter_fn,
    const std::vector<u32>& matching_ids,
    float selectivity) {
    
    if (matching_ids.empty()) {
        return {};
    }
    
    // --- 1. Get entry point from metadata ---
    u32 enter_point_id;
    u32 max_l;
    u32 node_count;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            node_count = meta_guard->node_count.load();
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    if (enter_point_id == HNSWMetadataPage::invalid_node_id || node_count == 0) {
        return {};
    }

    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

    // --- 2. Dynamic ef_search scaling based on selectivity ---
    // Lower selectivity needs larger ef_search to maintain recall
    // ACORN uses γ = 1/selectivity as expansion factor
    size_t base_ef = (ef_search_param == 0) ? std::max(static_cast<size_t>(efConstruction), k) : ef_search_param;
    
    // Scale ef_search inversely with selectivity, capped to avoid excessive computation
    // For 10% selectivity: gamma=10, sqrt(10)≈3.16, so ef_search ≈ base_ef * 3.16
    // We use a more aggressive scaling to ensure good recall at medium selectivity
    float gamma = std::min(100.0f, 1.0f / std::max(0.001f, selectivity));
    // Use linear scaling for medium selectivity (>5%) for better recall
    size_t ef_search;
    if (selectivity >= 0.05f) {
        // Medium selectivity: more aggressive scaling
        ef_search = std::min(static_cast<size_t>(base_ef * gamma), static_cast<size_t>(5000));
    } else {
        // Low selectivity: sqrt scaling to avoid excessive computation  
        ef_search = std::min(static_cast<size_t>(base_ef * std::sqrt(gamma)), static_cast<size_t>(3000));
    }
    ef_search = std::max(ef_search, k * 4);
    
    VisitedList* visited_nodes = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = visited_nodes->mass;
    vl_type visited_array_tag = visited_nodes->curV;
    
    // Results tracking
    std::priority_queue<std::pair<float, u32>> top_candidates;  // For graph traversal
    std::priority_queue<std::pair<float, u32>> filtered_results; // Final filtered results (max-heap)
    std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, 
                        std::greater<std::pair<float, u32>>> candidate_queue;  // Min-heap for BFS
    
    // --- 3. Initialize with multiple entry points ---
    // ACORN seeds with random entry points from the matching set
    // This helps when the standard HNSW entry point is far from the predicate subgraph
    
    // First, find entry point via standard HNSW traversal of upper layers
    float entry_dist;
    for (;;) {
        try {
            GuardORelaxed<HNSWPage> initial_guard(getNodePID(enter_point_id), index_array);
            NodeAccessor initial_acc(initial_guard.ptr, getNodeIndexInPage(enter_point_id), this);
            entry_dist = DistanceMetric::compare(query, initial_acc.getVector(), Dim);
        } catch (const OLCRestartException&) {
            continue;
        }
        break;
    }
    
    // Navigate through upper layers to find better entry point
    std::pair<float, u32> entry_point_for_layer0 = {entry_dist, enter_point_id};
    if (max_l > 0) {
        entry_point_for_layer0 = searchBaseLayer(query, enter_point_id, max_l, 1);
    }
    
    // Get visited list size for bounds checking - do this EARLY before any visited_array access
    u32 visited_array_size = visited_nodes->numelements;
    
    // Add HNSW entry point (with bounds check)
    if (entry_point_for_layer0.second < visited_array_size) {
        candidate_queue.push(entry_point_for_layer0);
        top_candidates.push(entry_point_for_layer0);
        visited_array[entry_point_for_layer0.second] = visited_array_tag;
        
        if (filter_fn(entry_point_for_layer0.second)) {
            filtered_results.push(entry_point_for_layer0);
        }
    } else {
        CALIBY_LOG_ERROR("HNSW", "ACORN entry_point out of bounds: id=", entry_point_for_layer0.second,
                       " visited_array_size=", visited_array_size);
    }
    
    // Add random entry points from matching set (ACORN-style seeding)
    // Sample more entry points for better coverage - especially important for medium selectivity
    // For 10% selectivity with 100k docs, we have ~10k matching docs
    // sqrt(10000) ≈ 100, which is a good number of seeds
    size_t num_random_seeds = std::min(
        static_cast<size_t>(std::sqrt(matching_ids.size()) * 2 + 10),
        std::min(matching_ids.size(), static_cast<size_t>(200))
    );
    
    // Use deterministic sampling based on query for reproducibility
    std::hash<float> hasher;
    size_t seed = hasher(query[0]) ^ (hasher(query[1]) << 1);
    std::mt19937 rng(seed);
    
    std::vector<size_t> sample_indices(matching_ids.size());
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    
    // Debug: Check first few elements of matching_ids for ALL calls
    // static int call_count = 0;
    // call_count++;
    // if (matching_ids.size() > 5) {
    //     CALIBY_LOG_WARN("HNSW", "ACORN matching_ids call #", call_count, ": size=", matching_ids.size(),
    //                    " first 5 elements: [0]=", matching_ids[0], " [1]=", matching_ids[1], 
    //                    " [2]=", matching_ids[2], " [3]=", matching_ids[3], " [4]=", matching_ids[4],
    //                    " data ptr=", reinterpret_cast<uintptr_t>(matching_ids.data()));
    // }
    
    for (size_t i = 0; i < num_random_seeds && i < sample_indices.size(); ++i) {
        std::uniform_int_distribution<size_t> dist(i, sample_indices.size() - 1);
        std::swap(sample_indices[i], sample_indices[dist(rng)]);
        
        size_t idx = sample_indices[i];
        if (idx >= matching_ids.size()) {
            CALIBY_LOG_ERROR("HNSW", "ACORN sample_indices[", i, "]=", idx, 
                           " >= matching_ids.size()=", matching_ids.size());
            continue;
        }
        
        // // Debug: On call #18, verify early elements are still valid
        // if (call_count == 18 && i == 3) {
        //     // Check multiple known-good elements
        //     CALIBY_LOG_WARN("HNSW", "ACORN i=3 check: [0]=", matching_ids[0], " [1]=", matching_ids[1],
        //                    " [2]=", matching_ids[2], " [1000]=", (matching_ids.size() > 1000 ? matching_ids[1000] : 0),
        //                    " [5000]=", (matching_ids.size() > 5000 ? matching_ids[5000] : 0),
        //                    " [10000]=", (matching_ids.size() > 10000 ? matching_ids[10000] : 0),
        //                    " [15000]=", (matching_ids.size() > 15000 ? matching_ids[15000] : 0));
        // }
        
        u32 seed_id = matching_ids[idx];
        
        // Bounds check for seed_id
        if (seed_id >= visited_array_size) {
            CALIBY_LOG_ERROR("HNSW", "ACORN seed_id out of bounds: seed_id=", seed_id, 
                           " visited_array_size=", visited_array_size,
                           " max_elements=", this->max_elements_,
                           " matching_ids.size=", matching_ids.size(),
                           " idx=", idx,
                           " matching_ids.data()[", idx, "]=", matching_ids.data()[idx]);
            continue;  // Skip invalid seed instead of crashing
        }
        
        if (visited_array[seed_id] == visited_array_tag) continue;
        
        float seed_dist;
        try {
            GuardORelaxed<HNSWPage> seed_guard(getNodePID(seed_id), index_array);
            NodeAccessor seed_acc(seed_guard.ptr, getNodeIndexInPage(seed_id), this);
            seed_dist = DistanceMetric::compare(query, seed_acc.getVector(), Dim);
        } catch (const OLCRestartException&) {
            continue;
        }
        
        visited_array[seed_id] = visited_array_tag;
        candidate_queue.push({seed_dist, seed_id});
        top_candidates.push({seed_dist, seed_id});
        
        if (filtered_results.size() < k) {
            filtered_results.push({seed_dist, seed_id});
        } else if (seed_dist < filtered_results.top().first) {
            filtered_results.pop();
            filtered_results.push({seed_dist, seed_id});
        }
    }
    
    // --- 4. ACORN-style beam search with 2-hop neighbor expansion ---
    while (!candidate_queue.empty()) {
        auto current_pair = candidate_queue.top();
        candidate_queue.pop();

        // Early termination: if current candidate is worse than worst in ef_search set
        if (top_candidates.size() >= ef_search && current_pair.first > top_candidates.top().first) {
            break;
        }
        
        u32 current_id = current_pair.second;
        bool current_passes_filter = filter_fn(current_id);
        
        try {
            GuardO<HNSWPage> current_page_guard(getNodePID(current_id), index_array);
            NodeAccessor current_acc(current_page_guard.ptr, getNodeIndexInPage(current_id), this);

            if (current_acc.getLevel() < 0) continue;

            auto neighbors = current_acc.getNeighbors(0, this);
            
            // Collect neighbor IDs for 2-hop expansion
            std::vector<u32> two_hop_candidates;
            
            for (size_t i = 0; i < neighbors.size(); ++i) {
                const u32 neighbor_id = neighbors[i];
                
                // Bounds check for neighbor_id
                if (neighbor_id >= visited_array_size) {
                    CALIBY_LOG_ERROR("HNSW", "ACORN neighbor_id out of bounds: neighbor_id=", neighbor_id, 
                                   " visited_array_size=", visited_array_size,
                                   " current_node=", current_pair.second);
                    continue;  // Skip invalid neighbor instead of crashing
                }
                
                if (visited_array[neighbor_id] == visited_array_tag) {
                    // Already visited - but may need for 2-hop
                    // If current doesn't pass filter, collect for 2-hop
                    if (!current_passes_filter) {
                        two_hop_candidates.push_back(neighbor_id);
                    }
                    continue;
                }
                visited_array[neighbor_id] = visited_array_tag;
                
                float neighbor_dist;
                try {
                    PID neighbor_pid = getNodePID(neighbor_id);
                    GuardORelaxed<HNSWPage> neighbor_page_guard(neighbor_pid, index_array);
                    NodeAccessor neighbor_acc(neighbor_page_guard.ptr, getNodeIndexInPage(neighbor_id), this);
                    neighbor_dist = DistanceMetric::compare(query, neighbor_acc.getVector(), Dim);
                } catch (const OLCRestartException&) {
                    i--;
                    continue;
                }

                // Add to graph traversal structures
                if (top_candidates.size() < ef_search || neighbor_dist < top_candidates.top().first) {
                    candidate_queue.push({neighbor_dist, neighbor_id});
                    top_candidates.push({neighbor_dist, neighbor_id});
                    if (top_candidates.size() > ef_search) {
                        top_candidates.pop();
                    }
                }
                
                // Add to filtered results if passes filter
                if (filter_fn(neighbor_id)) {
                    if (filtered_results.size() < k) {
                        filtered_results.push({neighbor_dist, neighbor_id});
                    } else if (neighbor_dist < filtered_results.top().first) {
                        filtered_results.pop();
                        filtered_results.push({neighbor_dist, neighbor_id});
                    }
                } else {
                    // Doesn't pass filter - candidate for 2-hop expansion
                    two_hop_candidates.push_back(neighbor_id);
                }
            }
            
            // --- 2-hop neighbor expansion (ACORN-1 style) ---
            // When a node doesn't pass the filter, explore its neighbors
            // This helps find paths through the predicate subgraph
            // Enable for selectivity <= 20% (covers the 10% benchmark case)
            if (selectivity <= 0.2f && !two_hop_candidates.empty()) {
                // Limit 2-hop expansion to avoid excessive computation
                size_t max_two_hop = std::min(two_hop_candidates.size(), static_cast<size_t>(M));
                
                for (size_t t = 0; t < max_two_hop; ++t) {
                    u32 two_hop_node = two_hop_candidates[t];
                    
                    try {
                        GuardO<HNSWPage> two_hop_guard(getNodePID(two_hop_node), index_array);
                        NodeAccessor two_hop_acc(two_hop_guard.ptr, getNodeIndexInPage(two_hop_node), this);
                        
                        if (two_hop_acc.getLevel() < 0) continue;
                        
                        auto two_hop_neighbors = two_hop_acc.getNeighbors(0, this);
                        
                        for (size_t j = 0; j < two_hop_neighbors.size(); ++j) {
                            u32 th_neighbor_id = two_hop_neighbors[j];
                            
                            // Bounds check for 2-hop neighbor
                            if (th_neighbor_id >= visited_array_size) {
                                CALIBY_LOG_ERROR("HNSW", "ACORN 2-hop neighbor out of bounds: id=", th_neighbor_id, 
                                               " visited_array_size=", visited_array_size);
                                continue;
                            }
                            
                            if (visited_array[th_neighbor_id] == visited_array_tag) continue;
                            
                            // OPTIMIZATION: Mark as visited BEFORE filter check to avoid
                            // re-checking the same non-matching nodes from different 2-hop paths
                            visited_array[th_neighbor_id] = visited_array_tag;
                            
                            // Only expand to nodes that pass filter (predicate subgraph traversal)
                            if (!filter_fn(th_neighbor_id)) continue;
                            
                            float th_dist;
                            try {
                                PID th_pid = getNodePID(th_neighbor_id);
                                GuardORelaxed<HNSWPage> th_guard(th_pid, index_array);
                                NodeAccessor th_acc(th_guard.ptr, getNodeIndexInPage(th_neighbor_id), this);
                                th_dist = DistanceMetric::compare(query, th_acc.getVector(), Dim);
                            } catch (const OLCRestartException&) {
                                j--;
                                continue;
                            }
                            
                            // Add to traversal structures
                            if (top_candidates.size() < ef_search || th_dist < top_candidates.top().first) {
                                candidate_queue.push({th_dist, th_neighbor_id});
                                top_candidates.push({th_dist, th_neighbor_id});
                                if (top_candidates.size() > ef_search) {
                                    top_candidates.pop();
                                }
                            }
                            
                            // Add to filtered results
                            if (filtered_results.size() < k) {
                                filtered_results.push({th_dist, th_neighbor_id});
                            } else if (th_dist < filtered_results.top().first) {
                                filtered_results.pop();
                                filtered_results.push({th_dist, th_neighbor_id});
                            }
                        }
                    } catch (const OLCRestartException&) {
                        continue;
                    }
                }
            }
            
        } catch (const OLCRestartException&) {
            candidate_queue.push(current_pair);
            continue;
        }
    }

    visited_list_pool_->releaseVisitedList(visited_nodes);

    // Extract filtered results sorted by distance ascending
    std::vector<std::pair<float, u32>> results;
    results.reserve(filtered_results.size());
    while (!filtered_results.empty()) {
        results.push_back(filtered_results.top());
        filtered_results.pop();
    }
    // print if filtered_results.size() < k for debugging
    if (results.size() < k) {
        std::cerr << "[DEBUG searchKnnFilteredACORN] Warning: only " << results.size() 
                  << " results found, fewer than requested k=" << k << std::endl;
    }
    std::reverse(results.begin(), results.end());
    
    return results;
}

template <typename DistanceMetric>
void HNSW<DistanceMetric>::addPoint(const float* point, u32& node_id_out) {
    u32 new_node_id;
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        new_node_id = meta_guard->node_count.fetch_add(1);
        if (new_node_id >= max_elements_) {
            meta_guard->node_count.fetch_sub(1);
            throw std::runtime_error("HNSW index is full.");
        }
        meta_guard->dirty = true;
    }
    node_id_out = new_node_id;
    addPoint_internal(point, new_node_id);
}

template <typename DistanceMetric>
void HNSW<DistanceMetric>::addPointWithId(const float* point, u32 node_id) {
    if (node_id >= max_elements_) {
        throw std::runtime_error("HNSW: node_id exceeds max_elements.");
    }
    {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        // Update node_count to be at least node_id + 1 (for proper page allocation tracking)
        u64 current_count = meta_guard->node_count.load();
        if (node_id >= current_count) {
            meta_guard->node_count.store(node_id + 1);
        }
        meta_guard->dirty = true;
    }
    addPoint_internal(point, node_id);
}

template <typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> HNSW<DistanceMetric>::searchKnn(const float* query, size_t k,
                                                                        size_t ef_search_param) {
    // --- 1. Get the global entry point ---
    u32 enter_point_id;
    u32 max_l;
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_l = meta_guard->max_level.load(std::memory_order_acquire);
            break;
        } catch (const OLCRestartException&) {
            continue; // Retry if metadata is being updated
        }
    }

    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        return {}; // Index is empty
    }

    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

    // --- 2. Calculate initial distance to entry point ---
    float entry_dist;
    for (;;) {
        try {
            GuardORelaxed<HNSWPage> initial_guard(getNodePID(enter_point_id), index_array);
            NodeAccessor initial_acc(initial_guard.ptr, getNodeIndexInPage(enter_point_id), this);
            if (stats) {
                entry_dist = this->calculateDistance(query, initial_acc.getVector());
            } else {
                entry_dist = DistanceMetric::compare(query, initial_acc.getVector(), Dim);
            }
            // Debug: Print the computed distance (uncomment for debugging)
            // std::cerr << "[DEBUG searchKnn] enter_point_id=" << enter_point_id 
            //           << " entry_dist=" << entry_dist 
            //           << " max_l=" << max_l << std::endl;
        } catch (const OLCRestartException&) {
            continue;
        }
        break;
    }

    // --- 3. Greedily search upper layers to find the best entry point for layer 0 ---
    std::pair<float, u32> entry_point_for_layer0 = {entry_dist, enter_point_id};
    
    if (max_l > 0) {
        // This call will find the best entry for layer 1 and return its ID and distance.
        entry_point_for_layer0 = searchBaseLayer<stats>(query, enter_point_id, max_l, 1);
    }
    
    // --- 3. Set search parameters ---
    size_t ef_search = (ef_search_param == 0) ? std::max(static_cast<size_t>(efConstruction), k) : ef_search_param;
    ef_search = std::max(ef_search, k);

    // --- 4. Perform beam search on the base layer (layer 0) ---
    // Pass the pre-calculated distance and ID to searchLayer to avoid redundant computation.
    auto candidates = searchLayer<stats>(query, entry_point_for_layer0.second, 0, ef_search, entry_point_for_layer0);

    // --- 5. Finalize and return results ---
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    // Remap BFS-internal IDs back to original (insertion-order) IDs
    if (!new_to_old_.empty()) {
        for (auto& p : candidates)
            p.second = internal_to_external(p.second);
    }
    return candidates;
}

template <typename DistanceMetric>
template <bool stats>
std::vector<std::vector<std::pair<float, u32>>> HNSW<DistanceMetric>::searchKnn_parallel(
    std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads) {
    if (queries.empty()) return {};
    if (queries.size() % Dim != 0) {
        throw std::invalid_argument("Total number of floats in query span is not a multiple of the vector dimension.");
    }
    size_t num_queries = queries.size() / Dim;
    size_t threads_to_use = (num_threads == 0) ? std::thread::hardware_concurrency() : num_threads;
    threads_to_use = std::min(threads_to_use, num_queries);
    std::vector<std::vector<std::pair<float, u32>>> all_results(num_queries);
    if (threads_to_use <= 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            all_results[i] = searchKnn<stats>(queries.data() + i * Dim, k, ef_search_param);
        }
    } else {
        // Reuse thread pool
        ThreadPool* pool = getOrCreateSearchPool(threads_to_use);
        std::vector<std::future<void>> futures;
        futures.reserve(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            futures.emplace_back(
                pool->enqueue([this, &all_results, queries_data = queries.data(), i, k, ef_search_param]() {
                    all_results[i] = this->searchKnn<stats>(queries_data + i * Dim, k, ef_search_param);
                }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    return all_results;
}

template <typename DistanceMetric>
u64 HNSW<DistanceMetric>::getBufferManagerAllocCount() const {
    return bm.allocCount.load();
}

template <typename DistanceMetric>
HNSWStats HNSW<DistanceMetric>::getStats() const {
    return stats_;
}

template <typename DistanceMetric>
std::string HNSW<DistanceMetric>::getIndexInfo() const {
    std::ostringstream info;

    info << "=== HNSW Index Information ===\n";

    // Configuration parameters
    info << "Configuration Parameters:\n";
    info << "  Vector Dimension: " << Dim << "\n";
    info << "  Max Neighbors (M): " << M << " (layers > 0)\n";
    info << "  Max Neighbors (M0): " << M0 << " (layer 0)\n";
    info << "  Construction ef: " << efConstruction << "\n";
    info << "  Max Levels: " << MaxLevel << "\n";

    // Memory layout information
    info << "\nMemory Layout:\n";
    info << "  Node Vector Size: " << VectorSize << " bytes\n";
    info << "  Max Neighbors Header Size: " << MaxNeighborsHeaderSize << " bytes\n";
    info << "  Max Neighbors List Size: " << MaxNeighborsListSize << " bytes\n";
    info << "  Fixed Node Size: " << FixedNodeSize << " bytes\n";
    info << "  Nodes Per Page: " << NodesPerPage << "\n";
    info << "  Page Size: " << pageSize << " bytes\n";

    // Runtime metadata from HNSWMetadataPage
    u32 enter_point_id = HNSWMetadataPage::invalid_node_id;
    u32 max_level = 0;
    u64 node_count = 0;
    u64 max_elements = 0;
    PID base_pid_value = 0;

    // Read metadata with retry loop for OLC
    for (;;) {
        try {
            GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
            enter_point_id = meta_guard->enter_point_node_id;
            max_level = meta_guard->max_level.load(std::memory_order_acquire);
            node_count = meta_guard->node_count.load(std::memory_order_acquire);
            max_elements = meta_guard->max_elements;
            base_pid_value = meta_guard->base_pid;
            break;
        } catch (const OLCRestartException&) {
            continue;
        }
    }

    info << "\nIndex Metadata:\n";
    info << "  Metadata PID: " << metadata_pid << "\n";
    info << "  Base PID: " << base_pid_value << "\n";
    info << "  Max Elements: " << max_elements << "\n";
    info << "  Current Node Count: " << node_count << "\n";
    info << "  Current Max Level: " << max_level << "\n";
    info << "  Entry Point Node ID: ";
    if (enter_point_id == HNSWMetadataPage::invalid_node_id) {
        info << "NONE (index is empty)\n";
    } else {
        info << enter_point_id << "\n";
    }

    // Entry point node information (if exists)
    if (enter_point_id != HNSWMetadataPage::invalid_node_id) {
        try {
            GuardO<HNSWPage> entry_page_guard(getNodePID(enter_point_id));
            NodeAccessor entry_acc(entry_page_guard.ptr, getNodeIndexInPage(enter_point_id), this);

            info << "\nEntry Point Node Details:\n";
            info << "  Node Level: " << entry_acc.getLevel() << "\n";
            info << "  Node PID: " << getNodePID(enter_point_id) << "\n";
            info << "  Node Index: " << getNodeIndexInPage(enter_point_id) << "\n";

            // Show neighbor counts per level
            for (u32 level = 0; level <= entry_acc.getLevel() && level < MaxLevel; ++level) {
                auto neighbors = entry_acc.getNeighbors(level, this);
                info << "  Level " << level << " neighbors: " << neighbors.size() << "\n";
            }
        } catch (const OLCRestartException&) {
            info << "\nEntry Point Node Details: (could not read due to concurrent access)\n";
        }
    }

    // Buffer manager statistics
    info << "\nBuffer Manager Statistics:\n";
    info << "  Total Allocations: " << bm.allocCount.load() << "\n";
    info << "  Physical Pages Used: " << bm.physUsedCount.load() << "\n";
    info << "  Read Operations: " << bm.readCount.load() << "\n";
    info << "  Write Operations: " << bm.writeCount.load() << "\n";

    // Calculate storage efficiency
    double storage_utilization = 0.0;
    if (max_elements > 0) {
        storage_utilization = (double)node_count / max_elements * 100.0;
    }
    info << "\nStorage Utilization:\n";
    info << "  Index Capacity Usage: " << std::fixed << std::setprecision(2) << storage_utilization << "%\n";

    u64 total_allocated_pages = (max_elements + NodesPerPage - 1) / NodesPerPage;
    if (total_allocated_pages > 0) {
        double page_utilization =
            (double)((node_count + NodesPerPage - 1) / NodesPerPage) / total_allocated_pages * 100.0;
        info << "  Page Utilization: " << std::fixed << std::setprecision(2) << page_utilization << "%\n";
    }

    info << "\n=== End Index Information ===\n";

    return info.str();
}

template <typename DistanceMetric>
void HNSW<DistanceMetric>::optimize_layout() {
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);
    u64 node_count = 0;

    // Get entry point and node count from metadata
    u32 entry_point = 0;
    try {
        GuardO<HNSWMetadataPage> meta_guard(metadata_pid);
        node_count = meta_guard->node_count.load(std::memory_order_acquire);
        entry_point = meta_guard->enter_point_node_id;
    } catch (const OLCRestartException&) { return; }
    if (node_count == 0) return;

    // Phase 1: BFS ordering
    std::vector<u32> new_to_old(node_count);
    std::vector<int32_t> old_to_new(node_count, -1);
    std::vector<u8> visited(node_count, 0);
    std::vector<u32> queue; queue.reserve(node_count);
    u32 bfs_count = 0;

    if (entry_point < node_count) { queue.push_back(entry_point); visited[entry_point] = 1; }
    for (size_t qh = 0; qh < queue.size(); ++qh) {
        u32 cur = queue[qh];
        new_to_old[bfs_count] = cur;
        old_to_new[cur] = static_cast<int32_t>(bfs_count);
        ++bfs_count;
        try {
            GuardORelaxed<HNSWPage> pg(getNodePID(cur), index_array);
            NodeAccessor acc(pg.ptr, getNodeIndexInPage(cur), this);
            auto nb = acc.getNeighbors(0, this);
            for (u32 nid : nb)
                if (nid < node_count && !visited[nid]) { visited[nid] = 1; queue.push_back(nid); }
        } catch (const OLCRestartException&) {}
    }
    for (u32 i = 0; i < node_count; ++i)
        if (!visited[i]) { new_to_old[bfs_count] = i; old_to_new[i] = static_cast<int32_t>(bfs_count); ++bfs_count; }

    // Save copies for later phases (new_to_old will be moved to member)
    auto n2o_copy = new_to_old;  // Phase 4 needs this
    new_to_old_ = std::move(new_to_old);

    // Phase 2: Read all nodes into temp buffer
    u8* buf = static_cast<u8*>(std::malloc(node_count * FixedNodeSize));
    if (!buf) return;
    for (u32 old_id = 0; old_id < node_count; ++old_id) {
        for (;;) {
            try {
                GuardO<HNSWPage> pg(getNodePID(old_id), index_array);
                u32 idx = getNodeIndexInPage(old_id);
                memcpy(buf + static_cast<u64>(old_id) * FixedNodeSize,
                       pg.ptr->getNodeData() + idx * FixedNodeSize, FixedNodeSize);
                break;
            } catch (const OLCRestartException&) {}
        }
    }

    // Phase 3: Remap all neighbor IDs (old → new)
    u64 remapped_count = 0;
    for (u32 old_id = 0; old_id < node_count; ++old_id) {
        u8* node = buf + static_cast<u64>(old_id) * FixedNodeSize;
        u16* counts = reinterpret_cast<u16*>(node + VectorSize);
        const u32* nb_all = reinterpret_cast<const u32*>(node + VectorSize + MaxNeighborsHeaderSize);
        for (u32 lv = 0; lv < MaxLevel; ++lv) {
            u16 n = counts[lv];
            if (n == 0) continue;
            u32 off = (lv == 0) ? 0 : (M0 + (lv - 1) * static_cast<u32>(M));
            u32* nb = const_cast<u32*>(nb_all + off);
            for (u16 j = 0; j < n; ++j) {
                u32 old_nb = nb[j];
                if (old_nb < node_count && old_to_new[old_nb] >= 0) {
                    nb[j] = static_cast<u32>(old_to_new[old_nb]);
                    ++remapped_count;
                }
            }
        }
    }

    // Phase 4: Write back in BFS order
    for (u32 new_pos = 0; new_pos < node_count; ++new_pos) {
        u32 old_id = n2o_copy[new_pos];
        PID pid = getNodePID(new_pos);
        u32 idx = getNodeIndexInPage(new_pos);
        GuardX<HNSWPage> pg(pid);
        memcpy(pg.ptr->getNodeData() + idx * FixedNodeSize,
               buf + static_cast<u64>(old_id) * FixedNodeSize, FixedNodeSize);
        pg.ptr->dirty = true;
    }

    // Update entry point
    if (entry_point < node_count && old_to_new[entry_point] >= 0) {
        GuardX<HNSWMetadataPage> meta_guard(metadata_pid);
        meta_guard->enter_point_node_id = static_cast<u32>(old_to_new[entry_point]);
        meta_guard->dirty = true;
    }

    // Verify: read back a few nodes and check neighbor IDs are valid
    u32 bad_refs = 0, total_refs = 0;
    for (u32 pos = 0; pos < std::min<u32>(10, node_count); ++pos) {
        try {
            GuardORelaxed<HNSWPage> pg(getNodePID(pos), index_array);
            NodeAccessor acc(pg.ptr, getNodeIndexInPage(pos), this);
            auto nb = acc.getNeighbors(0, this);
            for (u32 nid : nb) { ++total_refs; if (nid >= node_count) ++bad_refs; }
        } catch (...) {}
    }
    CALIBY_LOG_INFO("HNSW", "BFS done: ", bfs_count, " nodes, ", remapped_count, " remapped, verify: ", bad_refs, "/", total_refs, " bad refs");

    std::free(buf);
    CALIBY_LOG_INFO("HNSW", "BFS layout optimized for ", node_count, " nodes");
}

// --- Explicit Template Instantiation ---
// L2 Distance
template class HNSW<hnsw_distance::SIMDAcceleratedL2>;

// Explicit instantiation for templated member functions
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchKnn<true>(const float* query, size_t k, size_t ef_search_param);
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchKnn<false>(const float* query, size_t k, size_t ef_search_param);
template std::vector<std::vector<std::pair<float, u32>>> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchKnn_parallel<true>(std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads);
template std::vector<std::vector<std::pair<float, u32>>> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchKnn_parallel<false>(std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads);

// Explicit instantiation for internal templated helper methods
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchLayer<true>(const float* query, u32 entry_point_id, u32 level, size_t ef, std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchLayer<false>(const float* query, u32 entry_point_id, u32 level, size_t ef, std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedL2>::findBestEntryPointForLevel<true>(const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedL2>::findBestEntryPointForLevel<false>(const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchBaseLayer<true>(const float* query, u32 entry_point_id, int start_level, int end_level);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedL2>::searchBaseLayer<false>(const float* query, u32 entry_point_id, int start_level, int end_level);

// Inner Product Distance
template class HNSW<hnsw_distance::SIMDAcceleratedIP>;

template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchKnn<true>(const float* query, size_t k, size_t ef_search_param);
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchKnn<false>(const float* query, size_t k, size_t ef_search_param);
template std::vector<std::vector<std::pair<float, u32>>> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchKnn_parallel<true>(std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads);
template std::vector<std::vector<std::pair<float, u32>>> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchKnn_parallel<false>(std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads);

template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchLayer<true>(const float* query, u32 entry_point_id, u32 level, size_t ef, std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchLayer<false>(const float* query, u32 entry_point_id, u32 level, size_t ef, std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedIP>::findBestEntryPointForLevel<true>(const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedIP>::findBestEntryPointForLevel<false>(const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchBaseLayer<true>(const float* query, u32 entry_point_id, int start_level, int end_level);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedIP>::searchBaseLayer<false>(const float* query, u32 entry_point_id, int start_level, int end_level);

// Cosine Distance
template class HNSW<hnsw_distance::SIMDAcceleratedCosine>;

template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchKnn<true>(const float* query, size_t k, size_t ef_search_param);
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchKnn<false>(const float* query, size_t k, size_t ef_search_param);
template std::vector<std::vector<std::pair<float, u32>>> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchKnn_parallel<true>(std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads);
template std::vector<std::vector<std::pair<float, u32>>> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchKnn_parallel<false>(std::span<const float> queries, size_t k, size_t ef_search_param, size_t num_threads);

template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchLayer<true>(const float* query, u32 entry_point_id, u32 level, size_t ef, std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::vector<std::pair<float, u32>> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchLayer<false>(const float* query, u32 entry_point_id, u32 level, size_t ef, std::optional<std::pair<float, u32>> initial_entry_dist_pair);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedCosine>::findBestEntryPointForLevel<true>(const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedCosine>::findBestEntryPointForLevel<false>(const float* query, u32 entry_point_id, int level, float entry_point_dist);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchBaseLayer<true>(const float* query, u32 entry_point_id, int start_level, int end_level);
template std::pair<float, u32> HNSW<hnsw_distance::SIMDAcceleratedCosine>::searchBaseLayer<false>(const float* query, u32 entry_point_id, int start_level, int end_level);