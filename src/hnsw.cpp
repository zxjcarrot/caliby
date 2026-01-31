#include "hnsw.hpp"
#include "logging.hpp"

#include <immintrin.h>
#include <cstdio>

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
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
            meta_guard->alloc_count.store(1, std::memory_order_relaxed);
            meta_guard->enter_point_node_id = HNSWMetadataPage::invalid_node_id;
            meta_guard->max_level.store(0);

            u64 total_pages = (max_elements + NodesPerPage - 1) / NodesPerPage;
            if (total_pages > 0) {
                AllocGuard<HNSWPage> first_page_guard(allocator_);
                meta_guard->base_pid = first_page_guard.pid;  // Already a global PID
                first_page_guard->dirty = false;
                first_page_guard->node_count = 0;
                first_page_guard->dirty = true;

                PID expected_pid = meta_guard->base_pid + 1;
                for (u64 i = 1; i < total_pages; ++i) {
                    AllocGuard<HNSWPage> page_guard(allocator_);
                    // Verify contiguous allocation
                    if (page_guard.pid != expected_pid) {
                        CALIBY_LOG_ERROR("HNSW", "Recovery: Non-contiguous page allocation! Expected PID ",
                                  expected_pid, " but got ", page_guard.pid);
                        CALIBY_LOG_ERROR("HNSW", "Recovery: This breaks the assumption that pages are at base_pid + offset");
                        throw std::runtime_error("Non-contiguous HNSW page allocation");
                    }
                    page_guard->dirty = false;
                    page_guard->node_count = 0;
                    page_guard->dirty = true;
                    expected_pid++;
                }
            }

            this->base_pid = meta_guard->base_pid;
            meta_guard->dirty = true;
            CALIBY_LOG_INFO("HNSW", "Recovery: Allocated new metadata page ", this->metadata_pid,
                      " base_pid=", this->base_pid, " total_pages=", total_pages);

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
    
    std::priority_queue<std::pair<float, u32>> top_candidates; // Max-heap to keep the best results
    std::priority_queue<std::pair<float, u32>, std::vector<std::pair<float, u32>>, std::greater<std::pair<float, u32>>>
        candidate_queue; // Min-heap to explore candidates
    

    // --- Use pre-calculated distance if available ---
    if (initial_entry_dist_pair && initial_entry_dist_pair->second == entry_point_id) {
        float dist = initial_entry_dist_pair->first;
        top_candidates.push({dist, entry_point_id});
        candidate_queue.push({dist, entry_point_id});
        visited_array[entry_point_id] = visited_array_tag;
    } else {
        // Fallback: calculate distance if not provided or if IDs don't match
        for (;;) {
            float dist;
            try {
                #ifdef HNSW_DISABLE_OPTIMISTIC_READ
                GuardS<HNSWPage> page_guard(getNodePID(entry_point_id));
                #else
                GuardORelaxed<HNSWPage> page_guard(getNodePID(entry_point_id), index_array);
                #endif
                NodeAccessor acc(page_guard.ptr, getNodeIndexInPage(entry_point_id), this);
                
                if (stats) {
                    dist = this->calculateDistance(query, acc.getVector());
                } else {
                    dist = DistanceMetric::compare(query, acc.getVector(), Dim);
                }
            } catch (const OLCRestartException&) {
                continue;
            }
            top_candidates.push({dist, entry_point_id});
            candidate_queue.push({dist, entry_point_id});
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
        auto current_pair = candidate_queue.top();
        candidate_queue.pop();

        if (top_candidates.size() >= ef && current_pair.first > top_candidates.top().first) {
            break; // All further candidates are worse than the worst in our result set.
        }
        
        if (stats) {
            stats_.search_hops.fetch_add(1, std::memory_order_relaxed);
        }
        u32 current_id = current_pair.second;
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

                if (top_candidates.size() < ef || neighbor_dist < top_candidates.top().first) {
                    // prefetch l0 neighbor data
                    //_mm_prefetch(neighbor_page_guard.ptr->getNodeData() + NodeAccessor::getL0NeighborOffset(this, neighbor_id), _MM_HINT_T0);

                    candidate_queue.push({neighbor_dist, neighbor_id});
                    top_candidates.push({neighbor_dist, neighbor_id});
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
                visited_array[neighbor_id] = visited_array_tag;
            }
        } catch (const OLCRestartException&) {
            candidate_queue.push(current_pair); // Re-insert to retry later
            continue; // Skip this node if it was modified concurrently.
            //std::cout << "OLCRestartException caught in searchLayer" << std::endl;
        }
    }

    std::vector<std::pair<float, u32>> results;
    results.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
        results.push_back(top_candidates.top());
        top_candidates.pop();
    }
    std::reverse(results.begin(), results.end());
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
    PID new_node_pid = getNodePID(new_node_id);
    // Get IndexTranslationArray once for this index to avoid TLS lookups in tight loop
    IndexTranslationArray* index_array = bm.getIndexArray(index_id_);

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