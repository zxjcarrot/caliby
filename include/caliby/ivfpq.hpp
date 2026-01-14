#ifndef IVFPQ_HPP
#define IVFPQ_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <shared_mutex>
#include <span>
#include <vector>

#include "calico.hpp"
#include "distance.hpp"

// Type aliases for distance metrics (matching HNSW patterns)
using L2Distance = hnsw_distance::SIMDAcceleratedL2;
// For inner product, we'd need to add it to distance.hpp. For now, alias to L2:
using InnerProductDistance = hnsw_distance::SIMDAcceleratedL2;  // TODO: implement actual inner product

// Forward declarations
class ThreadPool;

// --- IVF+PQ Configuration Constants ---
static constexpr u32 IVFPQ_MAX_CACHED_PAGES = 16;  // Max pages cached in InvListEntry for prefetching
static constexpr u32 IVFPQ_DEFAULT_RETRAIN_INTERVAL = 10000;  // Retrain centroids every N insertions
static constexpr u32 IVFPQ_DEFAULT_NUM_CLUSTERS = 256;  // Default K
static constexpr u32 IVFPQ_DEFAULT_NUM_SUBQUANTIZERS = 8;  // Default M for PQ
static constexpr u32 IVFPQ_BITS_PER_CODE = 8;  // 256 codes per subquantizer
static constexpr u32 IVFPQ_NUM_CODES = 256;  // 2^8 codes
static constexpr u32 IVFPОС_KMEANS_ITERATIONS = 25;  // K-means iterations

// --- IVF+PQ Stats ---
struct IVFPQStats {
    std::atomic<u64> dist_comps{0};
    std::atomic<u64> lists_probed{0};
    std::atomic<u64> vectors_scanned{0};
    
    u32 num_clusters = 0;
    u32 num_subquantizers = 0;
    std::vector<u64> list_sizes;
    double avg_list_size = 0.0;
    
    IVFPQStats() = default;
    
    IVFPQStats(const IVFPQStats& other) {
        dist_comps.store(other.dist_comps.load());
        lists_probed.store(other.lists_probed.load());
        vectors_scanned.store(other.vectors_scanned.load());
        num_clusters = other.num_clusters;
        num_subquantizers = other.num_subquantizers;
        list_sizes = other.list_sizes;
        avg_list_size = other.avg_list_size;
    }
    
    IVFPQStats& operator=(const IVFPQStats& other) {
        if (this != &other) {
            dist_comps.store(other.dist_comps.load());
            lists_probed.store(other.lists_probed.load());
            vectors_scanned.store(other.vectors_scanned.load());
            num_clusters = other.num_clusters;
            num_subquantizers = other.num_subquantizers;
            list_sizes = other.list_sizes;
            avg_list_size = other.avg_list_size;
        }
        return *this;
    }
    
    void reset_live_counters() {
        dist_comps.store(0);
        lists_probed.store(0);
        vectors_scanned.store(0);
    }
    
    std::string toString() const;
};

// --- Page Structures ---

// Use IVFPQMetaInfoCompact from calico.hpp for recovery metadata
// Alias for convenience
using IVFPQMetaInfo = IVFPQMetaInfoCompact;

// Detailed metadata page
struct IVFPQMetadataPage {
    bool dirty;
    u8 _padding1[7];
    
    u32 dim;
    u32 num_clusters;
    u32 num_subquantizers;
    u32 subvector_dim;
    u32 retrain_interval;
    u32 _padding2;
    
    u64 max_elements;
    std::atomic<u64> num_vectors;
    std::atomic<u64> last_train_count;
    std::atomic<u8> is_trained;
    u8 _padding3[7];
    
    PID centroids_base_pid;
    PID invlist_dir_base_pid;
    PID codebook_base_pid;
    
    u32 centroids_per_page;
    u32 entries_per_invlist_page;
};

// Centroid page - stores cluster centers
struct CentroidPage {
    bool dirty;
    u32 centroid_count;
    u8 padding[4];
    
    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(centroid_count) + sizeof(padding);
    
    float* getCentroidData() { return reinterpret_cast<float*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const float* getCentroidData() const { return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(this) + HeaderSize); }
    
    float* getCentroid(u32 idx, u32 dim) { return getCentroidData() + idx * dim; }
    const float* getCentroid(u32 idx, u32 dim) const { return getCentroidData() + idx * dim; }
};

// Inverted list entry - metadata for one cluster's list
struct InvListEntry {
    std::atomic<u32> list_size;          // Number of vectors in this list
    std::atomic<u32> num_pages;          // Total pages in linked list
    PID first_page_pid;                  // First page of inverted list
    PID last_page_pid;                   // Last page for fast append
    PID cached_pages[IVFPQ_MAX_CACHED_PAGES];  // First 16 pages for prefetching
    std::atomic<u32> cached_page_count;  // Number of valid cached pages
    
    void init() {
        list_size.store(0, std::memory_order_relaxed);
        num_pages.store(0, std::memory_order_relaxed);
        first_page_pid = BufferManager::invalidPID;
        last_page_pid = BufferManager::invalidPID;
        cached_page_count.store(0, std::memory_order_relaxed);
        for (u32 i = 0; i < IVFPQ_MAX_CACHED_PAGES; ++i) {
            cached_pages[i] = BufferManager::invalidPID;
        }
    }
};

// Inverted list directory page - contains InvListEntry for multiple clusters
struct InvListDirPage {
    bool dirty;
    u32 entry_count;
    u32 first_cluster_id;  // First cluster ID on this page
    u8 padding[4];
    
    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(entry_count) + sizeof(first_cluster_id) + sizeof(padding);
    
    InvListEntry* getEntries() { return reinterpret_cast<InvListEntry*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const InvListEntry* getEntries() const { return reinterpret_cast<const InvListEntry*>(reinterpret_cast<const u8*>(this) + HeaderSize); }
    
    InvListEntry* getEntry(u32 local_idx) { return getEntries() + local_idx; }
    const InvListEntry* getEntry(u32 local_idx) const { return getEntries() + local_idx; }
    
    static constexpr u32 entriesPerPage() {
        return (pageSize - HeaderSize) / sizeof(InvListEntry);
    }
};

// PQ code entry - stores one vector's PQ codes and original ID
struct alignas(4) PQCodeEntry {
    u32 original_id;  // Original vector ID
    // PQ codes follow (M bytes for 8-bit codes)
    
    u8* getCodes() { return reinterpret_cast<u8*>(this) + sizeof(original_id); }
    const u8* getCodes() const { return reinterpret_cast<const u8*>(this) + sizeof(original_id); }
};

// Inverted list data page - stores PQ-encoded vectors
struct InvListDataPage {
    bool dirty;
    PID next_page;             // Next page in linked list
    u32 count;                 // Number of entries in this page
    u32 capacity;              // Max entries this page can hold
    u8 padding[4];
    
    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(next_page) + sizeof(count) + sizeof(capacity) + sizeof(padding);
    
    u8* getEntryData() { return reinterpret_cast<u8*>(this) + HeaderSize; }
    const u8* getEntryData() const { return reinterpret_cast<const u8*>(this) + HeaderSize; }
    
    // Entry size = sizeof(u32) + num_subquantizers
    PQCodeEntry* getEntry(u32 idx, u32 entry_size) {
        return reinterpret_cast<PQCodeEntry*>(getEntryData() + idx * entry_size);
    }
    const PQCodeEntry* getEntry(u32 idx, u32 entry_size) const {
        return reinterpret_cast<const PQCodeEntry*>(getEntryData() + idx * entry_size);
    }
    
    static u32 entriesPerPage(u32 entry_size) {
        return (pageSize - HeaderSize) / entry_size;
    }
};

// PQ Codebook page - stores partial codebook for one subquantizer
// Since 256 codes * subvec_dim floats may not fit in one page, 
// we split across multiple pages
struct PQCodebookPage {
    bool dirty;
    u32 subquantizer_id;      // Which subquantizer this page belongs to
    u32 subvector_dim;        // Dimension of each code vector
    u32 page_index;           // Which page of the codebook (0, 1, 2, ...)
    u32 start_code;           // First code index on this page
    u32 code_count;           // Number of codes on this page
    u8 padding[4];
    
    static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(subquantizer_id) + 
        sizeof(subvector_dim) + sizeof(page_index) + sizeof(start_code) + 
        sizeof(code_count) + sizeof(padding);
    
    float* getCodebook() { return reinterpret_cast<float*>(reinterpret_cast<u8*>(this) + HeaderSize); }
    const float* getCodebook() const { return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(this) + HeaderSize); }
    
    // Get code vector relative to this page's start
    float* getCode(u32 local_idx) { return getCodebook() + local_idx * subvector_dim; }
    const float* getCode(u32 local_idx) const { return getCodebook() + local_idx * subvector_dim; }
    
    // Calculate how many codes can fit per page given subvector dimension
    static u32 codesPerPage(u32 subvec_dim) {
        return (pageSize - HeaderSize) / (subvec_dim * sizeof(float));
    }
    
    // Calculate number of pages needed for full codebook
    static u32 pagesNeeded(u32 num_codes, u32 subvec_dim) {
        u32 codes_per_page = codesPerPage(subvec_dim);
        return (num_codes + codes_per_page - 1) / codes_per_page;
    }
};

// --- Main IVFPQ Class ---

template <typename DistanceMetric = L2Distance>
class IVFPQ {
public:
    // Constructor
    IVFPQ(u64 max_elements, size_t dim, u32 num_clusters = IVFPQ_DEFAULT_NUM_CLUSTERS,
          u32 num_subquantizers = IVFPQ_DEFAULT_NUM_SUBQUANTIZERS,
          u32 retrain_interval = IVFPQ_DEFAULT_RETRAIN_INTERVAL,
          bool skip_recovery = false, uint32_t index_id = 0, const std::string& name = "");
    
    ~IVFPQ();
    
    // Training
    void train(const float* training_vectors, u64 n_train, u32 kmeans_iters = IVFPОС_KMEANS_ITERATIONS);
    bool isTrained() const;
    
    // Adding vectors
    void addPoint(const float* vector, u32 vector_id);
    void addPoints(const float* vectors, const u32* ids, u64 count, size_t num_threads = 1);
    
    // Search
    template <bool stats = false>
    std::vector<std::pair<float, u32>> search(const float* query, size_t k, size_t nprobe);
    
    template <bool stats = false>
    std::vector<std::vector<std::pair<float, u32>>> searchBatch(
        std::span<const float> queries, size_t k, size_t nprobe, size_t num_threads = 1);
    
    // Statistics and info
    IVFPQStats getStats() const;
    void resetStats() { stats_.reset_live_counters(); }
    
    const std::string& getName() const { return name_; }
    u64 size() const;
    u32 getDim() const { return dim_; }
    u32 getNumClusters() const { return num_clusters_; }
    u32 getNumSubquantizers() const { return num_subquantizers_; }
    
    // Persistence
    void flush();

private:
    // Configuration
    uint32_t index_id_;
    std::string name_;
    PIDAllocator* allocator_;
    
    u32 dim_;
    u32 num_clusters_;
    u32 num_subquantizers_;
    u32 subvector_dim_;
    u32 retrain_interval_;
    u64 max_elements_;
    
    // Page locations
    PID metadata_pid_;
    PID centroids_base_pid_;
    PID invlist_dir_base_pid_;
    PID codebook_base_pid_;
    
    // Computed sizes
    u32 centroids_per_page_;
    u32 centroid_pages_;
    u32 entries_per_dir_page_;
    u32 dir_pages_;
    u32 codes_per_codebook_page_;   // How many codes fit per page
    u32 codebook_pages_per_subq_;   // Pages needed per subquantizer
    u32 pq_entry_size_;  // sizeof(u32) + num_subquantizers_
    u32 entries_per_invlist_page_;
    u32 codebook_pages_;  // One per subquantizer typically
    
    // Runtime state
    std::atomic<bool> is_trained_{false};
    bool recovered_from_disk_ = false;
    mutable IVFPQStats stats_;
    
    // Thread pools
    mutable std::unique_ptr<ThreadPool> search_thread_pool_{nullptr};
    mutable size_t search_pool_size_{0};
    mutable std::mutex search_pool_mutex_;
    
    mutable std::unique_ptr<ThreadPool> add_thread_pool_{nullptr};
    mutable size_t add_pool_size_{0};
    mutable std::mutex add_pool_mutex_;
    
    // In-memory cache for fast access during operations
    mutable std::vector<float> centroids_cache_;
    mutable std::vector<std::vector<float>> codebook_cache_;  // [M][256 * subvector_dim]
    mutable std::vector<std::vector<float>> codebook_norms_cache_;  // [M][256] - precomputed ||codebook[i]||^2
    mutable std::shared_mutex cache_mutex_;
    mutable bool cache_valid_ = false;
    
    // Helper methods
    ThreadPool* getOrCreateSearchPool(size_t num_threads) const;
    ThreadPool* getOrCreateAddPool(size_t num_threads) const;
    
    // Distance computation
    inline float computeDistance(const float* v1, const float* v2) const {
        return DistanceMetric::compare(v1, v2, dim_);
    }
    
    inline float computeSubvectorDistance(const float* v1, const float* v2, u32 subvec_dim) const {
        return DistanceMetric::compare(v1, v2, subvec_dim);
    }
    
    // K-means helpers
    void kmeansStep(const float* vectors, u64 n, float* centroids, u32 k, std::vector<u32>& assignments);
    void initializeCentroidsKMeansPlusPlus(const float* vectors, u64 n, float* centroids, u32 k);
    
    // PQ helpers
    void trainPQCodebooks(const float* vectors, u64 n);
    void trainSingleCodebook(u32 m, const float* vectors, u64 n, std::vector<float>& codebook);
    void encodeVector(const float* vector, u8* codes) const;
    void encodeResidual(const float* vector, const float* centroid, u8* codes) const;
    float computeADCDistance(const float* query, const u8* codes, const std::vector<std::vector<float>>& distance_tables) const;
    void precomputeDistanceTables(const float* query, std::vector<std::vector<float>>& tables) const;
    
    // Batch operations (SIMD optimized)
    void findNearestCentroidsBatch(const float* vectors, u64 count, u32* cluster_ids) const;
    void encodeVectorsBatch(const float* vectors, u64 count, u8* codes) const;
    void computeResidualsBatch(const float* vectors, const u32* cluster_ids, u64 count, float* residuals) const;
    
    // Centroid operations
    u32 findNearestCentroid(const float* vector) const;
    std::vector<std::pair<float, u32>> findNearestCentroids(const float* vector, size_t nprobe) const;
    
    // Cache management
    void loadCachesFromDisk() const;
    void invalidateCache();
    
    // Inverted list operations
    void appendToInvList(u32 cluster_id, u32 vector_id, const u8* pq_codes);
    
    // Page helpers
    inline PID getClusterDirPage(u32 cluster_id) const {
        return invlist_dir_base_pid_ + (cluster_id / entries_per_dir_page_);
    }
    
    inline u32 getClusterLocalIdx(u32 cluster_id) const {
        return cluster_id % entries_per_dir_page_;
    }
    
    inline PID getCentroidPage(u32 centroid_id) const {
        return centroids_base_pid_ + (centroid_id / centroids_per_page_);
    }
    
    inline u32 getCentroidLocalIdx(u32 centroid_id) const {
        return centroid_id % centroids_per_page_;
    }
    
    // Online retraining
    void maybeRetrain();
    void updateCentroids();
};

// Explicit instantiation declarations
extern template class IVFPQ<L2Distance>;
// Note: InnerProductDistance is currently aliased to L2Distance
// Once a proper inner product implementation exists, uncomment below:
// extern template class IVFPQ<InnerProductDistance>;

#endif // IVFPQ_HPP
