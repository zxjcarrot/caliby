#include "ivfpq.hpp"
#include "logging.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

// ============================================================================
// SIMD AVX2 Distance Computation (inspired by FAISS)
// ============================================================================

#ifdef __AVX2__

namespace {

// Horizontal sum of __m128 (4 floats)
inline float horizontal_sum_128(__m128 v) {
    __m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
    __m128 v1 = _mm_add_ps(v, v0);
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 v3 = _mm_add_ps(v1, v2);
    return _mm_cvtss_f32(v3);
}

// Horizontal sum of __m256 (8 floats)
inline float horizontal_sum_256(__m256 v) {
    __m128 v0 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    return horizontal_sum_128(v0);
}

// Compute distance for a single code with M=8, ksub=256
// Uses AVX2 gather instruction to fetch all 8 distance values at once
inline float distance_single_code_m8(const float* dis_table, const uint8_t* code) {
    constexpr size_t ksub = 256;
    
    // Create offsets: [0*256, 1*256, 2*256, ..., 7*256]
    __m256i offsets = _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
    
    // Load 8 uint8 codes and convert to int32
    // We need to load 8 bytes - use _mm_loadl_epi64 for 64-bit load
    __m128i codes_128 = _mm_loadl_epi64((const __m128i*)code);
    
    // Convert 8 uint8 values to 8 int32 values
    __m256i indices = _mm256_cvtepu8_epi32(codes_128);
    
    // Add offsets to get actual table indices
    __m256i table_indices = _mm256_add_epi32(indices, offsets);
    
    // Gather 8 float values from distance table
    __m256 distances = _mm256_i32gather_ps(dis_table, table_indices, sizeof(float));
    
    // Horizontal sum
    return horizontal_sum_256(distances);
}

// Compute distances for 4 codes at once with M=8, ksub=256
// This hides memory latency by processing multiple codes in parallel
inline void distance_four_codes_m8(
    const float* dis_table,
    const uint8_t* __restrict code0,
    const uint8_t* __restrict code1,
    const uint8_t* __restrict code2,
    const uint8_t* __restrict code3,
    float& dist0, float& dist1, float& dist2, float& dist3) {
    
    // Create offsets: [0*256, 1*256, 2*256, ..., 7*256]
    __m256i offsets = _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
    
    // Load 8 uint8 codes for each of 4 vectors
    __m128i codes0 = _mm_loadl_epi64((const __m128i*)code0);
    __m128i codes1 = _mm_loadl_epi64((const __m128i*)code1);
    __m128i codes2 = _mm_loadl_epi64((const __m128i*)code2);
    __m128i codes3 = _mm_loadl_epi64((const __m128i*)code3);
    
    // Convert and gather for all 4 codes
    __m256i idx0 = _mm256_cvtepu8_epi32(codes0);
    __m256i idx1 = _mm256_cvtepu8_epi32(codes1);
    __m256i idx2 = _mm256_cvtepu8_epi32(codes2);
    __m256i idx3 = _mm256_cvtepu8_epi32(codes3);
    
    idx0 = _mm256_add_epi32(idx0, offsets);
    idx1 = _mm256_add_epi32(idx1, offsets);
    idx2 = _mm256_add_epi32(idx2, offsets);
    idx3 = _mm256_add_epi32(idx3, offsets);
    
    __m256 dists0 = _mm256_i32gather_ps(dis_table, idx0, sizeof(float));
    __m256 dists1 = _mm256_i32gather_ps(dis_table, idx1, sizeof(float));
    __m256 dists2 = _mm256_i32gather_ps(dis_table, idx2, sizeof(float));
    __m256 dists3 = _mm256_i32gather_ps(dis_table, idx3, sizeof(float));
    
    // Horizontal sums
    dist0 = horizontal_sum_256(dists0);
    dist1 = horizontal_sum_256(dists1);
    dist2 = horizontal_sum_256(dists2);
    dist3 = horizontal_sum_256(dists3);
}

// Generalized SIMD distance computation for any M value
// Processes subquantizers in chunks of 8 using AVX2 gather
inline float distance_single_code_general(
    const float* dis_table, const uint8_t* code, uint32_t num_subquantizers) {
    
    constexpr size_t ksub = 256;
    float total_dist = 0.0f;
    
    // Process in chunks of 8 subquantizers
    uint32_t m = 0;
    for (; m + 8 <= num_subquantizers; m += 8) {
        // Offsets for this chunk: [m*256, (m+1)*256, ..., (m+7)*256]
        __m256i offsets = _mm256_setr_epi32(
            m * ksub, (m + 1) * ksub, (m + 2) * ksub, (m + 3) * ksub,
            (m + 4) * ksub, (m + 5) * ksub, (m + 6) * ksub, (m + 7) * ksub);
        
        // Load 8 codes starting at position m
        __m128i codes_128 = _mm_loadl_epi64((const __m128i*)(code + m));
        __m256i indices = _mm256_cvtepu8_epi32(codes_128);
        __m256i table_indices = _mm256_add_epi32(indices, offsets);
        __m256 distances = _mm256_i32gather_ps(dis_table, table_indices, sizeof(float));
        total_dist += horizontal_sum_256(distances);
    }
    
    // Handle remaining subquantizers (less than 8) with scalar code
    for (; m < num_subquantizers; ++m) {
        total_dist += dis_table[m * ksub + code[m]];
    }
    
    return total_dist;
}

// Generalized 4-way SIMD distance for any M value
inline void distance_four_codes_general(
    const float* dis_table,
    const uint8_t* __restrict code0,
    const uint8_t* __restrict code1,
    const uint8_t* __restrict code2,
    const uint8_t* __restrict code3,
    uint32_t num_subquantizers,
    float& dist0, float& dist1, float& dist2, float& dist3) {
    
    constexpr size_t ksub = 256;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    
    // Process in chunks of 8 subquantizers
    uint32_t m = 0;
    for (; m + 8 <= num_subquantizers; m += 8) {
        __m256i offsets = _mm256_setr_epi32(
            m * ksub, (m + 1) * ksub, (m + 2) * ksub, (m + 3) * ksub,
            (m + 4) * ksub, (m + 5) * ksub, (m + 6) * ksub, (m + 7) * ksub);
        
        __m128i c0 = _mm_loadl_epi64((const __m128i*)(code0 + m));
        __m128i c1 = _mm_loadl_epi64((const __m128i*)(code1 + m));
        __m128i c2 = _mm_loadl_epi64((const __m128i*)(code2 + m));
        __m128i c3 = _mm_loadl_epi64((const __m128i*)(code3 + m));
        
        __m256i idx0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c0), offsets);
        __m256i idx1 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c1), offsets);
        __m256i idx2 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c2), offsets);
        __m256i idx3 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c3), offsets);
        
        acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(dis_table, idx0, sizeof(float)));
        acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(dis_table, idx1, sizeof(float)));
        acc2 = _mm256_add_ps(acc2, _mm256_i32gather_ps(dis_table, idx2, sizeof(float)));
        acc3 = _mm256_add_ps(acc3, _mm256_i32gather_ps(dis_table, idx3, sizeof(float)));
    }
    
    // Sum accumulators
    dist0 = horizontal_sum_256(acc0);
    dist1 = horizontal_sum_256(acc1);
    dist2 = horizontal_sum_256(acc2);
    dist3 = horizontal_sum_256(acc3);
    
    // Handle remaining subquantizers with scalar code
    for (; m < num_subquantizers; ++m) {
        dist0 += dis_table[m * ksub + code0[m]];
        dist1 += dis_table[m * ksub + code1[m]];
        dist2 += dis_table[m * ksub + code2[m]];
        dist3 += dis_table[m * ksub + code3[m]];
    }
}

// Specialized high-performance 4-way distance for M=32 (common case for SIFT/128-dim)
// FAISS-style: unrolled 4 iterations of M=8, with better instruction scheduling
inline void distance_four_codes_m32(
    const float* dis_table,
    const uint8_t* __restrict code0,
    const uint8_t* __restrict code1,
    const uint8_t* __restrict code2,
    const uint8_t* __restrict code3,
    float& dist0, float& dist1, float& dist2, float& dist3) {
    
    constexpr size_t ksub = 256;
    
    // Pre-compute all offset vectors (m=0,8,16,24)
    const __m256i off0 = _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
    const __m256i off8 = _mm256_setr_epi32(2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840);
    const __m256i off16 = _mm256_setr_epi32(4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888);
    const __m256i off24 = _mm256_setr_epi32(6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936);
    
    // Load all 32 bytes for each code at once (better cache efficiency)
    __m256i codes0_full = _mm256_loadu_si256((const __m256i*)code0);
    __m256i codes1_full = _mm256_loadu_si256((const __m256i*)code1);
    __m256i codes2_full = _mm256_loadu_si256((const __m256i*)code2);
    __m256i codes3_full = _mm256_loadu_si256((const __m256i*)code3);
    
    // Extract low 64-bits (bytes 0-7) for m=0..7
    __m128i c0_0 = _mm256_castsi256_si128(codes0_full);
    __m128i c1_0 = _mm256_castsi256_si128(codes1_full);
    __m128i c2_0 = _mm256_castsi256_si128(codes2_full);
    __m128i c3_0 = _mm256_castsi256_si128(codes3_full);
    
    // Process m=0..7
    __m256i idx0_0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c0_0), off0);
    __m256i idx1_0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c1_0), off0);
    __m256i idx2_0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c2_0), off0);
    __m256i idx3_0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c3_0), off0);
    
    __m256 acc0 = _mm256_i32gather_ps(dis_table, idx0_0, sizeof(float));
    __m256 acc1 = _mm256_i32gather_ps(dis_table, idx1_0, sizeof(float));
    __m256 acc2 = _mm256_i32gather_ps(dis_table, idx2_0, sizeof(float));
    __m256 acc3 = _mm256_i32gather_ps(dis_table, idx3_0, sizeof(float));
    
    // Extract bytes 8-15 for m=8..15
    __m128i c0_8 = _mm_srli_si128(c0_0, 8);
    __m128i c1_8 = _mm_srli_si128(c1_0, 8);
    __m128i c2_8 = _mm_srli_si128(c2_0, 8);
    __m128i c3_8 = _mm_srli_si128(c3_0, 8);
    
    // Process m=8..15  
    __m256i idx0_8 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c0_8), off8);
    __m256i idx1_8 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c1_8), off8);
    __m256i idx2_8 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c2_8), off8);
    __m256i idx3_8 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c3_8), off8);
    
    acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(dis_table, idx0_8, sizeof(float)));
    acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(dis_table, idx1_8, sizeof(float)));
    acc2 = _mm256_add_ps(acc2, _mm256_i32gather_ps(dis_table, idx2_8, sizeof(float)));
    acc3 = _mm256_add_ps(acc3, _mm256_i32gather_ps(dis_table, idx3_8, sizeof(float)));
    
    // Extract high 128 bits (bytes 16-31)
    __m128i c0_16 = _mm256_extracti128_si256(codes0_full, 1);
    __m128i c1_16 = _mm256_extracti128_si256(codes1_full, 1);
    __m128i c2_16 = _mm256_extracti128_si256(codes2_full, 1);
    __m128i c3_16 = _mm256_extracti128_si256(codes3_full, 1);
    
    // Process m=16..23
    __m256i idx0_16 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c0_16), off16);
    __m256i idx1_16 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c1_16), off16);
    __m256i idx2_16 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c2_16), off16);
    __m256i idx3_16 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c3_16), off16);
    
    acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(dis_table, idx0_16, sizeof(float)));
    acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(dis_table, idx1_16, sizeof(float)));
    acc2 = _mm256_add_ps(acc2, _mm256_i32gather_ps(dis_table, idx2_16, sizeof(float)));
    acc3 = _mm256_add_ps(acc3, _mm256_i32gather_ps(dis_table, idx3_16, sizeof(float)));
    
    // Extract bytes 24-31 for m=24..31
    __m128i c0_24 = _mm_srli_si128(c0_16, 8);
    __m128i c1_24 = _mm_srli_si128(c1_16, 8);
    __m128i c2_24 = _mm_srli_si128(c2_16, 8);
    __m128i c3_24 = _mm_srli_si128(c3_16, 8);
    
    // Process m=24..31
    __m256i idx0_24 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c0_24), off24);
    __m256i idx1_24 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c1_24), off24);
    __m256i idx2_24 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c2_24), off24);
    __m256i idx3_24 = _mm256_add_epi32(_mm256_cvtepu8_epi32(c3_24), off24);
    
    acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(dis_table, idx0_24, sizeof(float)));
    acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(dis_table, idx1_24, sizeof(float)));
    acc2 = _mm256_add_ps(acc2, _mm256_i32gather_ps(dis_table, idx2_24, sizeof(float)));
    acc3 = _mm256_add_ps(acc3, _mm256_i32gather_ps(dis_table, idx3_24, sizeof(float)));
    
    // Final horizontal sums
    dist0 = horizontal_sum_256(acc0);
    dist1 = horizontal_sum_256(acc1);
    dist2 = horizontal_sum_256(acc2);
    dist3 = horizontal_sum_256(acc3);
}

} // anonymous namespace

#endif // __AVX2__

// ============================================================================
// SIMD-optimized batch operations for addPoints
// ============================================================================

#ifdef __AVX2__

namespace {

// Compute L2 distance between two vectors using AVX2
inline float simd_l2_distance(const float* a, const float* b, uint32_t dim) {
    __m256 sum = _mm256_setzero_ps();
    
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    float result = horizontal_sum_256(sum);
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    
    return result;
}

// Compute residuals for a batch of vectors: residual[i] = vector[i] - centroid[cluster_ids[i]]
inline void compute_residuals_batch(
    const float* vectors,
    const float* centroids,
    const uint32_t* cluster_ids,
    float* residuals,
    uint64_t count,
    uint32_t dim) {
    
    #pragma omp parallel for if(count > 1000)
    for (uint64_t i = 0; i < count; ++i) {
        const float* vec = vectors + i * dim;
        const float* centroid = centroids + cluster_ids[i] * dim;
        float* residual = residuals + i * dim;
        
        uint32_t d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(vec + d);
            __m256 c = _mm256_loadu_ps(centroid + d);
            __m256 r = _mm256_sub_ps(v, c);
            _mm256_storeu_ps(residual + d, r);
        }
        for (; d < dim; ++d) {
            residual[d] = vec[d] - centroid[d];
        }
    }
}

// FAISS-style: Compute distance table and find argmin in one pass
// This is much faster than computing all distances then finding min
// Uses: c[i] = ||x - codebook[i]||^2 = ||x||^2 + ||codebook[i]||^2 - 2*<x, codebook[i]>
// Since ||x||^2 is constant, we compute ||codebook[i]||^2 - 2*<x, codebook[i]> and find argmin
inline uint8_t find_nearest_codebook_entry(
    const float* subvec,           // subvector to encode, size dsub
    const float* codebook,         // codebook entries, size 256 * dsub
    const float* codebook_norms,   // precomputed ||codebook[i]||^2, size 256
    uint32_t dsub) {
    
    constexpr uint32_t ksub = 256;
    
    // Compute x_norm = ||x||^2 (constant, not needed for argmin)
    // We compute: dis[i] = codebook_norms[i] - 2 * dot(x, codebook[i])
    // which is equivalent to L2 distance up to constant ||x||^2
    
    float min_dist = std::numeric_limits<float>::max();
    uint8_t best_code = 0;
    
    // Process 4 codebook entries at a time for better ILP
    uint32_t c = 0;
    for (; c + 4 <= ksub; c += 4) {
        // Compute dot products with 4 codebook entries
        __m256 dot0 = _mm256_setzero_ps();
        __m256 dot1 = _mm256_setzero_ps();
        __m256 dot2 = _mm256_setzero_ps();
        __m256 dot3 = _mm256_setzero_ps();
        
        const float* cb0 = codebook + c * dsub;
        const float* cb1 = codebook + (c + 1) * dsub;
        const float* cb2 = codebook + (c + 2) * dsub;
        const float* cb3 = codebook + (c + 3) * dsub;
        
        uint32_t d = 0;
        for (; d + 8 <= dsub; d += 8) {
            __m256 x = _mm256_loadu_ps(subvec + d);
            dot0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(cb0 + d), dot0);
            dot1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(cb1 + d), dot1);
            dot2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(cb2 + d), dot2);
            dot3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(cb3 + d), dot3);
        }
        
        // Horizontal sum
        float dp0 = horizontal_sum_256(dot0);
        float dp1 = horizontal_sum_256(dot1);
        float dp2 = horizontal_sum_256(dot2);
        float dp3 = horizontal_sum_256(dot3);
        
        // Handle tail
        for (; d < dsub; ++d) {
            float x = subvec[d];
            dp0 += x * cb0[d];
            dp1 += x * cb1[d];
            dp2 += x * cb2[d];
            dp3 += x * cb3[d];
        }
        
        // dis = norm - 2*dot (effectively L2 distance minus ||x||^2)
        float dis0 = codebook_norms[c] - 2.0f * dp0;
        float dis1 = codebook_norms[c + 1] - 2.0f * dp1;
        float dis2 = codebook_norms[c + 2] - 2.0f * dp2;
        float dis3 = codebook_norms[c + 3] - 2.0f * dp3;
        
        if (dis0 < min_dist) { min_dist = dis0; best_code = c; }
        if (dis1 < min_dist) { min_dist = dis1; best_code = c + 1; }
        if (dis2 < min_dist) { min_dist = dis2; best_code = c + 2; }
        if (dis3 < min_dist) { min_dist = dis3; best_code = c + 3; }
    }
    
    // Handle remaining
    for (; c < ksub; ++c) {
        const float* cb = codebook + c * dsub;
        float dot = 0.0f;
        for (uint32_t d = 0; d < dsub; ++d) {
            dot += subvec[d] * cb[d];
        }
        float dis = codebook_norms[c] - 2.0f * dot;
        if (dis < min_dist) {
            min_dist = dis;
            best_code = c;
        }
    }
    
    return best_code;
}

// Precompute codebook norms for faster encoding
inline void compute_codebook_norms(
    const float* codebook,
    uint32_t dsub,
    float* norms) {
    
    constexpr uint32_t ksub = 256;
    
    for (uint32_t c = 0; c < ksub; ++c) {
        const float* cb = codebook + c * dsub;
        __m256 sum = _mm256_setzero_ps();
        
        uint32_t d = 0;
        for (; d + 8 <= dsub; d += 8) {
            __m256 v = _mm256_loadu_ps(cb + d);
            sum = _mm256_fmadd_ps(v, v, sum);
        }
        
        float norm = horizontal_sum_256(sum);
        for (; d < dsub; ++d) {
            norm += cb[d] * cb[d];
        }
        norms[c] = norm;
    }
}

} // anonymous namespace

#endif // __AVX2__

// Simple thread pool for parallel operations
class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::invoke_result<F>::type> {
        using return_type = typename std::invoke_result<F>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// --- IVFPQStats ---
std::string IVFPQStats::toString() const {
    std::stringstream ss;
    ss << "IVF+PQ Stats:\n";
    ss << "  Distance computations: " << dist_comps.load() << "\n";
    ss << "  Lists probed: " << lists_probed.load() << "\n";
    ss << "  Vectors scanned: " << vectors_scanned.load() << "\n";
    ss << "  Num clusters: " << num_clusters << "\n";
    ss << "  Num subquantizers: " << num_subquantizers << "\n";
    ss << "  Avg list size: " << std::fixed << std::setprecision(2) << avg_list_size << "\n";
    return ss.str();
}

// --- Helper: Thread Pool Access ---
template <typename DistanceMetric>
ThreadPool* IVFPQ<DistanceMetric>::getOrCreateSearchPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(search_pool_mutex_);
    if (!search_thread_pool_ || search_pool_size_ != num_threads) {
        search_thread_pool_.reset(new ThreadPool(num_threads));
        search_pool_size_ = num_threads;
    }
    return search_thread_pool_.get();
}

template <typename DistanceMetric>
ThreadPool* IVFPQ<DistanceMetric>::getOrCreateAddPool(size_t num_threads) const {
    std::lock_guard<std::mutex> lock(add_pool_mutex_);
    if (!add_thread_pool_ || add_pool_size_ != num_threads) {
        add_thread_pool_.reset(new ThreadPool(num_threads));
        add_pool_size_ = num_threads;
    }
    return add_thread_pool_.get();
}

// --- Constructor ---
template <typename DistanceMetric>
IVFPQ<DistanceMetric>::IVFPQ(u64 max_elements, size_t dim, u32 num_clusters,
                              u32 num_subquantizers, u32 retrain_interval,
                              bool skip_recovery, uint32_t index_id, const std::string& name)
    : index_id_(index_id),
      name_(name),
      dim_(static_cast<u32>(dim)),
      num_clusters_(num_clusters),
      num_subquantizers_(num_subquantizers),
      retrain_interval_(retrain_interval),
      max_elements_(max_elements) {
    
    if (dim == 0) {
        throw std::runtime_error("IVFPQ dimension must be greater than zero.");
    }
    if (dim % num_subquantizers != 0) {
        throw std::runtime_error("IVFPQ: dimension must be divisible by num_subquantizers.");
    }
    
    subvector_dim_ = dim_ / num_subquantizers_;
    
    // Compute page layout sizes
    centroids_per_page_ = (pageSize - CentroidPage::HeaderSize) / (dim_ * sizeof(float));
    centroid_pages_ = (num_clusters_ + centroids_per_page_ - 1) / centroids_per_page_;
    
    entries_per_dir_page_ = InvListDirPage::entriesPerPage();
    dir_pages_ = (num_clusters_ + entries_per_dir_page_ - 1) / entries_per_dir_page_;
    
    pq_entry_size_ = sizeof(u32) + num_subquantizers_;  // ID + PQ codes
    // Align to 4 bytes
    pq_entry_size_ = (pq_entry_size_ + 3) & ~3;
    entries_per_invlist_page_ = InvListDataPage::entriesPerPage(pq_entry_size_);
    
    // Calculate codebook pages - may need multiple pages per subquantizer
    codes_per_codebook_page_ = PQCodebookPage::codesPerPage(subvector_dim_);
    codebook_pages_per_subq_ = PQCodebookPage::pagesNeeded(IVFPQ_NUM_CODES, subvector_dim_);
    codebook_pages_ = num_subquantizers_ * codebook_pages_per_subq_;
    
    CALIBY_LOG_INFO("IVFPQ", "Initialization: Dim=", dim_, ", K=", num_clusters_, ", M=", num_subquantizers_, ", SubvecDim=", subvector_dim_);
    CALIBY_LOG_INFO("IVFPQ", "  CentroidsPerPage=", centroids_per_page_, ", CentroidPages=", centroid_pages_);
    CALIBY_LOG_INFO("IVFPQ", "  EntriesPerDirPage=", entries_per_dir_page_, ", DirPages=", dir_pages_);
    CALIBY_LOG_INFO("IVFPQ", "  PQEntrySize=", pq_entry_size_, ", EntriesPerInvListPage=", entries_per_invlist_page_);
    CALIBY_LOG_INFO("IVFPQ", "  CodesPerCodebookPage=", codes_per_codebook_page_, ", CodebookPagesPerSubq=", codebook_pages_per_subq_, ", TotalCodebookPages=", codebook_pages_);
    CALIBY_LOG_INFO("IVFPQ", "  RetrainInterval=", retrain_interval_);
    
    // Calculate total pages needed
    u64 total_pages = 1 + centroid_pages_ + dir_pages_ + codebook_pages_;
    // Add estimated inverted list pages - each cluster gets at least one page,
    // plus additional pages for overflow (vectors per cluster / entries_per_page)
    u64 base_invlist_pages = num_clusters_;  // At least one page per cluster
    u64 overflow_pages = (max_elements / entries_per_invlist_page_) + num_clusters_;
    total_pages += base_invlist_pages + overflow_pages;
    // Add some extra headroom for safety
    total_pages = std::max(total_pages, (u64)(max_elements / 10 + 256));
    
    allocator_ = bm.getOrCreateAllocatorForIndex(index_id_, total_pages);
    
    // Compute global metadata page ID
    PID global_metadata_page_id;
    if (bm.supportsMultiIndexPIDs() && index_id_ > 0) {
        global_metadata_page_id = (static_cast<PID>(index_id_) << 32) | 0ULL;
    } else {
        global_metadata_page_id = 0;
    }
    
    // Try to recover or initialize fresh
    GuardX<MetaDataPage> meta_page_guard(global_metadata_page_id);
    MetaDataPage* meta_page_ptr = meta_page_guard.ptr;
    IVFPQMetaInfo* meta_info = reinterpret_cast<IVFPQMetaInfo*>(&meta_page_ptr->ivfpq_meta);
    
    const bool has_existing_meta = meta_info->isValid();
    const bool params_match = has_existing_meta &&
                              meta_info->dim == dim_ &&
                              meta_info->num_clusters == num_clusters_ &&
                              meta_info->num_subquantizers == num_subquantizers_;
    
    CALIBY_LOG_INFO("IVFPQ", "Recovery: skip_recovery=", (skip_recovery ? "true" : "false"), " has_existing_meta=", (has_existing_meta ? "true" : "false"));
    
    if (!skip_recovery && params_match) {
        // Recover existing index
        metadata_pid_ = meta_info->metadata_pid;
        centroids_base_pid_ = meta_info->centroids_base_pid;
        invlist_dir_base_pid_ = meta_info->invlist_dir_base_pid;
        codebook_base_pid_ = meta_info->codebook_base_pid;
        is_trained_.store(meta_info->is_trained != 0, std::memory_order_release);
        recovered_from_disk_ = true;
        
        CALIBY_LOG_INFO("IVFPQ", "Recovery: Recovered existing index. metadata_pid=", metadata_pid_, " is_trained=", is_trained_.load());
    } else {
        // Initialize fresh
        if (has_existing_meta) {
            meta_info->valid = 0;
            meta_page_guard->dirty = true;
            CALIBY_LOG_INFO("IVFPQ", "Recovery: Existing metadata invalidated for rebuild");
        }
        
        // Allocate metadata page
        AllocGuard<IVFPQMetadataPage> metadata_guard(allocator_);
        metadata_pid_ = metadata_guard.pid;
        metadata_guard->dirty = true;
        metadata_guard->dim = dim_;
        metadata_guard->num_clusters = num_clusters_;
        metadata_guard->num_subquantizers = num_subquantizers_;
        metadata_guard->subvector_dim = subvector_dim_;
        metadata_guard->retrain_interval = retrain_interval_;
        metadata_guard->max_elements = max_elements_;
        metadata_guard->num_vectors.store(0, std::memory_order_relaxed);
        metadata_guard->last_train_count.store(0, std::memory_order_relaxed);
        metadata_guard->is_trained.store(0, std::memory_order_relaxed);
        metadata_guard->centroids_per_page = centroids_per_page_;
        metadata_guard->entries_per_invlist_page = entries_per_invlist_page_;
        
        // Allocate centroid pages
        {
            AllocGuard<CentroidPage> first_centroid_guard(allocator_);
            centroids_base_pid_ = first_centroid_guard.pid;
            first_centroid_guard->dirty = true;
            first_centroid_guard->centroid_count = 0;
            
            for (u32 i = 1; i < centroid_pages_; ++i) {
                AllocGuard<CentroidPage> page_guard(allocator_);
                page_guard->dirty = true;
                page_guard->centroid_count = 0;
            }
        }
        
        // Allocate inverted list directory pages
        {
            AllocGuard<InvListDirPage> first_dir_guard(allocator_);
            invlist_dir_base_pid_ = first_dir_guard.pid;
            first_dir_guard->dirty = true;
            first_dir_guard->entry_count = std::min(entries_per_dir_page_, num_clusters_);
            first_dir_guard->first_cluster_id = 0;
            // Initialize entries
            for (u32 j = 0; j < first_dir_guard->entry_count; ++j) {
                first_dir_guard->getEntry(j)->init();
            }
            
            for (u32 i = 1; i < dir_pages_; ++i) {
                AllocGuard<InvListDirPage> page_guard(allocator_);
                page_guard->dirty = true;
                u32 start_cluster = i * entries_per_dir_page_;
                u32 count = std::min(entries_per_dir_page_, num_clusters_ - start_cluster);
                page_guard->entry_count = count;
                page_guard->first_cluster_id = start_cluster;
                for (u32 j = 0; j < count; ++j) {
                    page_guard->getEntry(j)->init();
                }
            }
        }
        
        // Allocate PQ codebook pages (multi-page per subquantizer)
        {
            for (u32 sq = 0; sq < num_subquantizers_; ++sq) {
                for (u32 page_idx = 0; page_idx < codebook_pages_per_subq_; ++page_idx) {
                    AllocGuard<PQCodebookPage> page_guard(allocator_);
                    if (sq == 0 && page_idx == 0) {
                        codebook_base_pid_ = page_guard.pid;
                    }
                    page_guard->dirty = true;
                    page_guard->subquantizer_id = sq;
                    page_guard->subvector_dim = subvector_dim_;
                    page_guard->page_index = page_idx;
                    page_guard->start_code = page_idx * codes_per_codebook_page_;
                    u32 codes_remaining = IVFPQ_NUM_CODES - page_idx * codes_per_codebook_page_;
                    page_guard->code_count = std::min(codes_per_codebook_page_, codes_remaining);
                }
            }
        }
        
        // Update metadata page with PIDs
        metadata_guard->centroids_base_pid = centroids_base_pid_;
        metadata_guard->invlist_dir_base_pid = invlist_dir_base_pid_;
        metadata_guard->codebook_base_pid = codebook_base_pid_;
        
        // Update global metadata
        meta_info->magic_value = IVFPQMetaInfo::magic;
        meta_info->valid = 1;
        meta_info->dim = dim_;
        meta_info->num_clusters = num_clusters_;
        meta_info->num_subquantizers = num_subquantizers_;
        meta_info->subvector_dim = subvector_dim_;
        meta_info->metadata_pid = metadata_pid_;
        meta_info->centroids_base_pid = centroids_base_pid_;
        meta_info->invlist_dir_base_pid = invlist_dir_base_pid_;
        meta_info->codebook_base_pid = codebook_base_pid_;
        meta_info->max_elements = max_elements_;
        meta_info->num_vectors.store(0, std::memory_order_relaxed);
        meta_info->last_train_count.store(0, std::memory_order_relaxed);
        meta_info->retrain_interval = retrain_interval_;
        meta_info->is_trained = 0;
        meta_page_guard->dirty = true;
        
        CALIBY_LOG_INFO("IVFPQ", "Recovery: Allocated new index. metadata_pid=", metadata_pid_, " centroids_base=", centroids_base_pid_, " invlist_dir_base=", invlist_dir_base_pid_, " codebook_base=", codebook_base_pid_);
    }
    
    // Print SIMD availability info
#ifdef __AVX2__
    #ifdef __AVX512F__
    CALIBY_LOG_DEBUG("IVFPQ", "SIMD: AVX2=enabled AVX512=enabled M=", num_subquantizers_, " (SIMD search ENABLED)");
    #else
    CALIBY_LOG_DEBUG("IVFPQ", "SIMD: AVX2=enabled AVX512=disabled M=", num_subquantizers_, " (SIMD search ENABLED)");
    #endif
#else
    #ifdef __AVX512F__
    CALIBY_LOG_DEBUG("IVFPQ", "SIMD: AVX2=disabled AVX512=enabled M=", num_subquantizers_, " (SIMD search DISABLED, no AVX2)");
    #else
    CALIBY_LOG_DEBUG("IVFPQ", "SIMD: AVX2=disabled AVX512=disabled M=", num_subquantizers_, " (SIMD search DISABLED, no AVX2)");
    #endif
#endif
}

// --- Destructor ---
template <typename DistanceMetric>
IVFPQ<DistanceMetric>::~IVFPQ() {
    search_thread_pool_.reset(nullptr);
    add_thread_pool_.reset(nullptr);
}

// --- K-means++ Initialization ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::initializeCentroidsKMeansPlusPlus(
    const float* vectors, u64 n, float* centroids, u32 k) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Choose first centroid randomly
    std::uniform_int_distribution<u64> dist(0, n - 1);
    u64 first_idx = dist(gen);
    std::memcpy(centroids, vectors + first_idx * dim_, dim_ * sizeof(float));
    
    // Distance from each point to nearest centroid
    std::vector<float> min_distances(n, std::numeric_limits<float>::max());
    
    for (u32 c = 1; c < k; ++c) {
        // Update distances to nearest centroid
        const float* last_centroid = centroids + (c - 1) * dim_;
        double sum_distances = 0.0;
        
        #pragma omp parallel for reduction(+:sum_distances) if(n > 10000)
        for (u64 i = 0; i < n; ++i) {
            float d = computeDistance(vectors + i * dim_, last_centroid);
            if (d < min_distances[i]) {
                min_distances[i] = d;
            }
            sum_distances += min_distances[i];
        }
        
        // Sample next centroid proportional to squared distance
        std::uniform_real_distribution<double> sample_dist(0.0, sum_distances);
        double target = sample_dist(gen);
        double cumsum = 0.0;
        u64 chosen_idx = 0;
        
        for (u64 i = 0; i < n; ++i) {
            cumsum += min_distances[i];
            if (cumsum >= target) {
                chosen_idx = i;
                break;
            }
        }
        
        std::memcpy(centroids + c * dim_, vectors + chosen_idx * dim_, dim_ * sizeof(float));
    }
}

// --- K-means Step ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::kmeansStep(
    const float* vectors, u64 n, float* centroids, u32 k, std::vector<u32>& assignments) {
    
    assignments.resize(n);
    
    // Assignment step
    #pragma omp parallel for if(n > 10000)
    for (u64 i = 0; i < n; ++i) {
        float min_dist = std::numeric_limits<float>::max();
        u32 best_cluster = 0;
        
        for (u32 c = 0; c < k; ++c) {
            float d = computeDistance(vectors + i * dim_, centroids + c * dim_);
            if (d < min_dist) {
                min_dist = d;
                best_cluster = c;
            }
        }
        assignments[i] = best_cluster;
    }
    
    // Update step
    std::vector<float> new_centroids(k * dim_, 0.0f);
    std::vector<u64> counts(k, 0);
    
    for (u64 i = 0; i < n; ++i) {
        u32 c = assignments[i];
        counts[c]++;
        for (u32 d = 0; d < dim_; ++d) {
            new_centroids[c * dim_ + d] += vectors[i * dim_ + d];
        }
    }
    
    for (u32 c = 0; c < k; ++c) {
        if (counts[c] > 0) {
            float inv_count = 1.0f / counts[c];
            for (u32 d = 0; d < dim_; ++d) {
                centroids[c * dim_ + d] = new_centroids[c * dim_ + d] * inv_count;
            }
        }
    }
}

// --- Train PQ Codebooks ---
// Helper function to train a single subquantizer's codebook
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::trainSingleCodebook(
    u32 m, const float* vectors, u64 n, std::vector<float>& codebook) {
    
    // Extract subvectors for this subquantizer
    std::vector<float> subvectors(n * subvector_dim_);
    u32 offset = m * subvector_dim_;
    
    for (u64 i = 0; i < n; ++i) {
        std::memcpy(subvectors.data() + i * subvector_dim_,
                   vectors + i * dim_ + offset,
                   subvector_dim_ * sizeof(float));
    }
    
    // Train codebook using k-means
    codebook.resize(IVFPQ_NUM_CODES * subvector_dim_);
    
    // K-means++ initialization for subvectors (use deterministic seed for reproducibility)
    {
        std::mt19937 gen(42 + m);  // Seed based on subquantizer index
        std::uniform_int_distribution<u64> dist(0, n - 1);
        
        // First centroid
        std::memcpy(codebook.data(), subvectors.data() + dist(gen) * subvector_dim_,
                   subvector_dim_ * sizeof(float));
        
        std::vector<float> min_dists(n, std::numeric_limits<float>::max());
        
        for (u32 c = 1; c < IVFPQ_NUM_CODES; ++c) {
            const float* last = codebook.data() + (c - 1) * subvector_dim_;
            double sum = 0.0;
            
            for (u64 i = 0; i < n; ++i) {
                float d = computeSubvectorDistance(subvectors.data() + i * subvector_dim_, last, subvector_dim_);
                if (d < min_dists[i]) min_dists[i] = d;
                sum += min_dists[i];
            }
            
            std::uniform_real_distribution<double> sample(0.0, sum);
            double target = sample(gen);
            double cumsum = 0.0;
            u64 chosen = 0;
            
            for (u64 i = 0; i < n; ++i) {
                cumsum += min_dists[i];
                if (cumsum >= target) { chosen = i; break; }
            }
            
            std::memcpy(codebook.data() + c * subvector_dim_,
                       subvectors.data() + chosen * subvector_dim_,
                       subvector_dim_ * sizeof(float));
        }
    }
    
    // K-means iterations (use 25 iterations like the coarse quantizer)
    std::vector<u32> assignments(n);
    for (u32 iter = 0; iter < 25; ++iter) {
        // Assignment - use FAISS-style optimized nearest neighbor search
        for (u64 i = 0; i < n; ++i) {
            const float* subvec = subvectors.data() + i * subvector_dim_;
            
#ifdef __AVX2__
            // Use precomputed norms optimization
            // First compute codebook norms if not done
            std::vector<float> cb_norms(IVFPQ_NUM_CODES);
            for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
                const float* cb = codebook.data() + c * subvector_dim_;
                float norm = 0.0f;
                for (u32 d = 0; d < subvector_dim_; ++d) {
                    norm += cb[d] * cb[d];
                }
                cb_norms[c] = norm;
            }
            
            // Find nearest using dot product optimization: dis = ||cb||^2 - 2*<subvec,cb>
            float min_d = std::numeric_limits<float>::max();
            u32 best = 0;
            
            // Process 4 codebook entries at a time
            u32 c = 0;
            for (; c + 4 <= IVFPQ_NUM_CODES; c += 4) {
                // Compute dot products
                float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f, dp3 = 0.0f;
                const float* cb0 = codebook.data() + c * subvector_dim_;
                const float* cb1 = cb0 + subvector_dim_;
                const float* cb2 = cb1 + subvector_dim_;
                const float* cb3 = cb2 + subvector_dim_;
                
                for (u32 d = 0; d < subvector_dim_; ++d) {
                    float v = subvec[d];
                    dp0 += v * cb0[d];
                    dp1 += v * cb1[d];
                    dp2 += v * cb2[d];
                    dp3 += v * cb3[d];
                }
                
                // Compute distances: ||cb||^2 - 2*<subvec,cb> (ignoring ||subvec||^2 for argmin)
                float d0 = cb_norms[c] - 2.0f * dp0;
                float d1 = cb_norms[c + 1] - 2.0f * dp1;
                float d2 = cb_norms[c + 2] - 2.0f * dp2;
                float d3 = cb_norms[c + 3] - 2.0f * dp3;
                
                if (d0 < min_d) { min_d = d0; best = c; }
                if (d1 < min_d) { min_d = d1; best = c + 1; }
                if (d2 < min_d) { min_d = d2; best = c + 2; }
                if (d3 < min_d) { min_d = d3; best = c + 3; }
            }
            
            // Handle remaining
            for (; c < IVFPQ_NUM_CODES; ++c) {
                float dp = 0.0f;
                const float* cb = codebook.data() + c * subvector_dim_;
                for (u32 d = 0; d < subvector_dim_; ++d) {
                    dp += subvec[d] * cb[d];
                }
                float dist = cb_norms[c] - 2.0f * dp;
                if (dist < min_d) { min_d = dist; best = c; }
            }
            assignments[i] = best;
#else
            // Fallback: naive distance computation
            float min_d = std::numeric_limits<float>::max();
            u32 best = 0;
            for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
                float d = computeSubvectorDistance(
                    subvec,
                    codebook.data() + c * subvector_dim_,
                    subvector_dim_);
                if (d < min_d) { min_d = d; best = c; }
            }
            assignments[i] = best;
#endif
        }
        
        // Update centroids
        std::vector<float> new_cb(IVFPQ_NUM_CODES * subvector_dim_, 0.0f);
        std::vector<u64> counts(IVFPQ_NUM_CODES, 0);
        
        for (u64 i = 0; i < n; ++i) {
            u32 c = assignments[i];
            counts[c]++;
            const float* sv = subvectors.data() + i * subvector_dim_;
            float* dst = new_cb.data() + c * subvector_dim_;
            for (u32 d = 0; d < subvector_dim_; ++d) {
                dst[d] += sv[d];
            }
        }
        
        for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                for (u32 d = 0; d < subvector_dim_; ++d) {
                    codebook[c * subvector_dim_ + d] = new_cb[c * subvector_dim_ + d] * inv;
                }
            }
        }
    }
}

template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::trainPQCodebooks(const float* vectors, u64 n) {
    CALIBY_LOG_INFO("IVFPQ", "Training PQ codebooks with ", n, " vectors (parallel)...");
    
    // Train all subquantizers in parallel (FAISS-style)
    std::vector<std::vector<float>> all_codebooks(num_subquantizers_);
    
    #pragma omp parallel for schedule(dynamic)
    for (u32 m = 0; m < num_subquantizers_; ++m) {
        trainSingleCodebook(m, vectors, n, all_codebooks[m]);
    }
    
    // Write all codebooks to disk (must be sequential for page allocation)
    for (u32 m = 0; m < num_subquantizers_; ++m) {
        const auto& codebook = all_codebooks[m];
        u32 codes_written = 0;
        for (u32 page_idx = 0; page_idx < codebook_pages_per_subq_; ++page_idx) {
            PID page_pid = codebook_base_pid_ + m * codebook_pages_per_subq_ + page_idx;
            GuardX<PQCodebookPage> cb_guard(page_pid);
            cb_guard->dirty = true;
            cb_guard->subquantizer_id = m;
            cb_guard->subvector_dim = subvector_dim_;
            cb_guard->page_index = page_idx;
            cb_guard->start_code = codes_written;
            
            u32 codes_remaining = IVFPQ_NUM_CODES - codes_written;
            u32 codes_this_page = std::min(codes_per_codebook_page_, codes_remaining);
            cb_guard->code_count = codes_this_page;
            
            std::memcpy(cb_guard->getCodebook(),
                       codebook.data() + codes_written * subvector_dim_,
                       codes_this_page * subvector_dim_ * sizeof(float));
            
            codes_written += codes_this_page;
        }
    }
    
    CALIBY_LOG_INFO("IVFPQ", "PQ codebook training complete.");
}

// --- Train ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::train(const float* training_vectors, u64 n_train, u32 kmeans_iters) {
    if (n_train < num_clusters_) {
        throw std::runtime_error("IVFPQ: Need at least num_clusters training vectors");
    }
    
    CALIBY_LOG_INFO("IVFPQ", "Training with ", n_train, " vectors, ", kmeans_iters, " iterations...");
    
    // Train coarse quantizer (cluster centroids)
    std::vector<float> centroids(num_clusters_ * dim_);
    
    // K-means++ initialization
    initializeCentroidsKMeansPlusPlus(training_vectors, n_train, centroids.data(), num_clusters_);
    
    // K-means iterations
    std::vector<u32> assignments;
    for (u32 iter = 0; iter < kmeans_iters; ++iter) {
        CALIBY_LOG_DEBUG("IVFPQ", "  K-means iteration ", iter + 1, "/", kmeans_iters);
        kmeansStep(training_vectors, n_train, centroids.data(), num_clusters_, assignments);
    }
    
    // Write centroids to disk
    for (u32 page_idx = 0; page_idx < centroid_pages_; ++page_idx) {
        GuardX<CentroidPage> page_guard(centroids_base_pid_ + page_idx);
        page_guard->dirty = true;
        
        u32 start_c = page_idx * centroids_per_page_;
        u32 end_c = std::min(start_c + centroids_per_page_, num_clusters_);
        page_guard->centroid_count = end_c - start_c;
        
        for (u32 c = start_c; c < end_c; ++c) {
            std::memcpy(page_guard->getCentroid(c - start_c, dim_),
                       centroids.data() + c * dim_,
                       dim_ * sizeof(float));
        }
    }
    
    // Compute residuals for PQ training (vector - centroid for each vector)
    std::vector<float> residuals(n_train * dim_);
    for (u64 i = 0; i < n_train; ++i) {
        u32 cluster_id = assignments[i];
        const float* centroid = centroids.data() + cluster_id * dim_;
        const float* vec = training_vectors + i * dim_;
        float* residual = residuals.data() + i * dim_;
        
        for (u32 d = 0; d < dim_; ++d) {
            residual[d] = vec[d] - centroid[d];
        }
    }
    
    // Train PQ codebooks on residuals
    trainPQCodebooks(residuals.data(), n_train);
    
    // Mark as trained
    is_trained_.store(true, std::memory_order_release);
    
    // Update metadata
    GuardX<IVFPQMetadataPage> meta_guard(metadata_pid_);
    meta_guard->dirty = true;
    meta_guard->is_trained.store(1, std::memory_order_release);
    meta_guard->last_train_count.store(0, std::memory_order_release);
    
    // Update global metadata page with trained status
    // For index_id=0, the global metadata page is at PID 0
    // For index_id>0, it's at (index_id << 32) | 0
    {
        PID global_metadata_page_id = (index_id_ > 0) 
            ? (static_cast<PID>(index_id_) << 32) | 0ULL 
            : 0ULL;
        GuardX<MetaDataPage> global_meta_guard(global_metadata_page_id);
        IVFPQMetaInfo* meta_info = reinterpret_cast<IVFPQMetaInfo*>(&global_meta_guard.ptr->ivfpq_meta);
        meta_info->is_trained = 1;
        global_meta_guard->dirty = true;
    }
    
    // Invalidate cache
    invalidateCache();
    
    CALIBY_LOG_INFO("IVFPQ", "Training complete.");
}

// --- Is Trained ---
template <typename DistanceMetric>
bool IVFPQ<DistanceMetric>::isTrained() const {
    return is_trained_.load(std::memory_order_acquire);
}

// --- Load Caches From Disk ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::loadCachesFromDisk() const {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (cache_valid_) return;
    
    // Load centroids
    centroids_cache_.resize(num_clusters_ * dim_);
    for (u32 page_idx = 0; page_idx < centroid_pages_; ++page_idx) {
        GuardO<CentroidPage> page_guard(centroids_base_pid_ + page_idx);
        u32 start_c = page_idx * centroids_per_page_;
        u32 count = page_guard->centroid_count;
        
        for (u32 i = 0; i < count; ++i) {
            std::memcpy(centroids_cache_.data() + (start_c + i) * dim_,
                       page_guard->getCentroid(i, dim_),
                       dim_ * sizeof(float));
        }
    }
    
    // Load codebooks (may span multiple pages)
    codebook_cache_.resize(num_subquantizers_);
    codebook_norms_cache_.resize(num_subquantizers_);
    
    for (u32 m = 0; m < num_subquantizers_; ++m) {
        codebook_cache_[m].resize(IVFPQ_NUM_CODES * subvector_dim_);
        codebook_norms_cache_[m].resize(IVFPQ_NUM_CODES);
        
        u32 codes_read = 0;
        for (u32 page_idx = 0; page_idx < codebook_pages_per_subq_; ++page_idx) {
            PID page_pid = codebook_base_pid_ + m * codebook_pages_per_subq_ + page_idx;
            GuardO<PQCodebookPage> cb_guard(page_pid);
            
            u32 codes_this_page = cb_guard->code_count;
            std::memcpy(codebook_cache_[m].data() + codes_read * subvector_dim_,
                       cb_guard->getCodebook(),
                       codes_this_page * subvector_dim_ * sizeof(float));
            
            codes_read += codes_this_page;
        }
        
        // Precompute codebook norms for faster encoding
#ifdef __AVX2__
        compute_codebook_norms(codebook_cache_[m].data(), subvector_dim_, codebook_norms_cache_[m].data());
#else
        for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
            float norm = 0.0f;
            const float* cb = codebook_cache_[m].data() + c * subvector_dim_;
            for (u32 d = 0; d < subvector_dim_; ++d) {
                norm += cb[d] * cb[d];
            }
            codebook_norms_cache_[m][c] = norm;
        }
#endif
    }
    
    cache_valid_ = true;
}

// --- Invalidate Cache ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::invalidateCache() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    cache_valid_ = false;
}

// --- Find Nearest Centroid ---
template <typename DistanceMetric>
u32 IVFPQ<DistanceMetric>::findNearestCentroid(const float* vector) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    float min_dist = std::numeric_limits<float>::max();
    u32 best_cluster = 0;
    
    for (u32 c = 0; c < num_clusters_; ++c) {
        float d = computeDistance(vector, centroids_cache_.data() + c * dim_);
        if (d < min_dist) {
            min_dist = d;
            best_cluster = c;
        }
    }
    
    return best_cluster;
}

// --- Find N Nearest Centroids ---
template <typename DistanceMetric>
std::vector<std::pair<float, u32>> IVFPQ<DistanceMetric>::findNearestCentroids(
    const float* vector, size_t nprobe) const {
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    // Min-heap to track top nprobe
    std::priority_queue<std::pair<float, u32>> max_heap;
    
    for (u32 c = 0; c < num_clusters_; ++c) {
        float d = computeDistance(vector, centroids_cache_.data() + c * dim_);
        
        if (max_heap.size() < nprobe) {
            max_heap.push({d, c});
        } else if (d < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({d, c});
        }
    }
    
    std::vector<std::pair<float, u32>> result;
    result.reserve(max_heap.size());
    while (!max_heap.empty()) {
        result.push_back(max_heap.top());
        max_heap.pop();
    }
    
    // Sort by distance ascending
    std::sort(result.begin(), result.end());
    return result;
}

// --- Encode Vector with PQ ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::encodeVector(const float* vector, u8* codes) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    for (u32 m = 0; m < num_subquantizers_; ++m) {
        const float* subvec = vector + m * subvector_dim_;
        const float* codebook = codebook_cache_[m].data();
        
        float min_dist = std::numeric_limits<float>::max();
        u8 best_code = 0;
        
        for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
            float d = computeSubvectorDistance(subvec, codebook + c * subvector_dim_, subvector_dim_);
            if (d < min_dist) {
                min_dist = d;
                best_code = static_cast<u8>(c);
            }
        }
        
        codes[m] = best_code;
    }
}

// --- Encode Residual with PQ (residual = vector - centroid) ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::encodeResidual(const float* vector, const float* centroid, u8* codes) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    // Compute and encode residual (vector - centroid) with PQ
    std::vector<float> residual(dim_);
    for (u32 d = 0; d < dim_; ++d) {
        residual[d] = vector[d] - centroid[d];
    }
    
    for (u32 m = 0; m < num_subquantizers_; ++m) {
        const float* subvec = residual.data() + m * subvector_dim_;
        const float* codebook = codebook_cache_[m].data();
        
        float min_dist = std::numeric_limits<float>::max();
        u8 best_code = 0;
        
        for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
            float d = computeSubvectorDistance(subvec, codebook + c * subvector_dim_, subvector_dim_);
            if (d < min_dist) {
                min_dist = d;
                best_code = static_cast<u8>(c);
            }
        }
        
        codes[m] = best_code;
    }
}

// --- Find Nearest Centroids Batch (SIMD optimized) ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::findNearestCentroidsBatch(
    const float* vectors, u64 count, u32* cluster_ids) const {
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    const float* centroids = centroids_cache_.data();
    
    // #pragma omp parallel for if(count > 100)
    for (u64 i = 0; i < count; ++i) {
        const float* vec = vectors + i * dim_;
        float min_dist = std::numeric_limits<float>::max();
        u32 best_cluster = 0;
        
#ifdef __AVX2__
        // SIMD-optimized distance computation
        for (u32 c = 0; c < num_clusters_; ++c) {
            const float* centroid = centroids + c * dim_;
            float dist = simd_l2_distance(vec, centroid, dim_);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
#else
        for (u32 c = 0; c < num_clusters_; ++c) {
            float d = computeDistance(vec, centroids + c * dim_);
            if (d < min_dist) {
                min_dist = d;
                best_cluster = c;
            }
        }
#endif
        cluster_ids[i] = best_cluster;
    }
}

// --- Encode Vectors Batch (SIMD optimized using FAISS-style norm trick) ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::encodeVectorsBatch(
    const float* vectors, u64 count, u8* codes) const {
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    // Process each vector - can be parallelized with OMP
    #pragma omp parallel for if(count > 100)
    for (u64 i = 0; i < count; ++i) {
        const float* vec = vectors + i * dim_;
        u8* vec_codes = codes + i * num_subquantizers_;
        
        for (u32 m = 0; m < num_subquantizers_; ++m) {
            const float* subvec = vec + m * subvector_dim_;
            const float* codebook = codebook_cache_[m].data();
            const float* codebook_norms = codebook_norms_cache_[m].data();
            
#ifdef __AVX2__
            // Use FAISS-style optimized encoding: find nearest via precomputed norms
            vec_codes[m] = find_nearest_codebook_entry(subvec, codebook, codebook_norms, subvector_dim_);
#else
            float min_dist = std::numeric_limits<float>::max();
            u8 best_code = 0;
            for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
                float d = computeSubvectorDistance(subvec, codebook + c * subvector_dim_, subvector_dim_);
                if (d < min_dist) {
                    min_dist = d;
                    best_code = static_cast<u8>(c);
                }
            }
            vec_codes[m] = best_code;
#endif
        }
    }
}

// --- Compute Residuals Batch (SIMD optimized) ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::computeResidualsBatch(
    const float* vectors, const u32* cluster_ids, u64 count, float* residuals) const {
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (!cache_valid_) {
        lock.unlock();
        loadCachesFromDisk();
        lock.lock();
    }
    
    const float* centroids = centroids_cache_.data();
    
#ifdef __AVX2__
    compute_residuals_batch(vectors, centroids, cluster_ids, residuals, count, dim_);
#else
    // #pragma omp parallel for if(count > 100)
    for (u64 i = 0; i < count; ++i) {
        const float* vec = vectors + i * dim_;
        const float* centroid = centroids + cluster_ids[i] * dim_;
        float* residual = residuals + i * dim_;
        for (u32 d = 0; d < dim_; ++d) {
            residual[d] = vec[d] - centroid[d];
        }
    }
#endif
}

// --- Compute ADC Distance ---
template <typename DistanceMetric>
float IVFPQ<DistanceMetric>::computeADCDistance(
    const float* query, const u8* codes,
    const std::vector<std::vector<float>>& distance_tables) const {
    
    float dist = 0.0f;
    for (u32 m = 0; m < num_subquantizers_; ++m) {
        dist += distance_tables[m][codes[m]];
    }
    return dist;
}

// --- Append to Inverted List ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::appendToInvList(u32 cluster_id, u32 vector_id, const u8* pq_codes) {
    if (cluster_id >= num_clusters_) {
        throw std::runtime_error("IVFPQ: Invalid cluster_id");
    }
    
    PID dir_page_pid = getClusterDirPage(cluster_id);
    u32 local_idx = getClusterLocalIdx(cluster_id);
    
    // Lock the directory entry
    GuardX<InvListDirPage> dir_guard(dir_page_pid);
    InvListEntry* entry = dir_guard->getEntry(local_idx);
    
    
    PID target_page_pid;
    bool need_new_page = false;
    
    if (entry->first_page_pid == BufferManager::invalidPID) {
        // First vector in this list
        need_new_page = true;
    } else {
        // Validate last_page_pid
        if (entry->last_page_pid == BufferManager::invalidPID) {
            // Invalid PID - reset the list
            entry->first_page_pid = BufferManager::invalidPID;
            need_new_page = true;
        } else {
            // Save last_page_pid 
            PID last_page_pid = entry->last_page_pid;
            
            // Check if last page has space
            GuardX<InvListDataPage> last_page_guard(last_page_pid);
            
            // NOTE: We don't validate capacity here because it may contain garbage
            // from BufferManager frame recycling. The actual capacity is compile-time
            // known as entries_per_invlist_page_, so we just check count.
            
            if (last_page_guard->count >= entries_per_invlist_page_) {
                // Page is full - need new page
                need_new_page = true;
            } else if (last_page_guard->count >= last_page_guard->capacity) {
                // Page is full - allocate new page
                need_new_page = true;
            } else {
                // Append to existing page
                target_page_pid = last_page_pid;
                PQCodeEntry* pq_entry = last_page_guard->getEntry(last_page_guard->count, pq_entry_size_);
                pq_entry->original_id = vector_id;
                std::memcpy(pq_entry->getCodes(), pq_codes, num_subquantizers_);
                last_page_guard->count++;
                last_page_guard->dirty = true;
            }
        }
    }
    
    if (need_new_page) {
        // Save previous last page PID if exists for linking
        PID prev_last_pid = BufferManager::invalidPID;
        if (entry->first_page_pid != BufferManager::invalidPID) {
            prev_last_pid = entry->last_page_pid;
        }
        
        // Allocate new page
        AllocGuard<InvListDataPage> new_page_guard(allocator_);
        PID new_page_pid = new_page_guard.pid;
        
        // CRITICAL FIX: Zero the ENTIRE page, not just the header!
        // Recycled frames contain garbage data from previous use
        std::memset(new_page_guard.ptr, 0, pageSize);
        
        // Set header fields  
        new_page_guard->next_page = BufferManager::invalidPID;
        new_page_guard->count = 1;
        new_page_guard->capacity = entries_per_invlist_page_;
        new_page_guard->dirty = true;
        
        // Write entry
        PQCodeEntry* pq_entry = new_page_guard->getEntry(0, pq_entry_size_);
        pq_entry->original_id = vector_id;
        std::memcpy(pq_entry->getCodes(), pq_codes, num_subquantizers_);
        
        
        // Link from previous page if exists
        if (prev_last_pid != BufferManager::invalidPID) {
            
            GuardX<InvListDataPage> prev_last_guard(prev_last_pid);
            prev_last_guard->next_page = new_page_pid;
            prev_last_guard->dirty = true;
            
            // Update directory entry
            entry->last_page_pid = new_page_pid;
            dir_guard->dirty = true;
        } else {
            // First page
            entry->first_page_pid = new_page_pid;
            entry->last_page_pid = new_page_pid;
            dir_guard->dirty = true;
        }
        
        // Update cached pages for prefetching
        u32 num_pages = entry->num_pages.load(std::memory_order_relaxed);
        if (num_pages < IVFPQ_MAX_CACHED_PAGES) {
            entry->cached_pages[num_pages] = new_page_pid;
            entry->cached_page_count.store(num_pages + 1, std::memory_order_release);
        }
        entry->num_pages.fetch_add(1, std::memory_order_relaxed);
    }
    
    entry->list_size.fetch_add(1, std::memory_order_relaxed);
    // Ensure directory page is marked dirty (may already be set above, but doesn't hurt)
    dir_guard->dirty = true;
}

// --- Add Point ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::addPoint(const float* vector, u32 vector_id) {
    if (!is_trained_.load(std::memory_order_acquire)) {
        throw std::runtime_error("IVFPQ: Index must be trained before adding vectors");
    }
    
    // Find nearest cluster
    u32 cluster_id = findNearestCentroid(vector);
    
    // Get centroid and compute residual (vector - centroid)
    std::vector<float> residual(dim_);
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            loadCachesFromDisk();
            lock.lock();
        }
        const float* centroid = centroids_cache_.data() + cluster_id * dim_;
        for (u32 d = 0; d < dim_; ++d) {
            residual[d] = vector[d] - centroid[d];
        }
    }
    
    // Encode residual with PQ
    std::vector<u8> pq_codes(num_subquantizers_);
    encodeVector(residual.data(), pq_codes.data());
    
    // Append to inverted list
    appendToInvList(cluster_id, vector_id, pq_codes.data());
    
    // Update count
    GuardX<IVFPQMetadataPage> meta_guard(metadata_pid_);
    u64 new_count = meta_guard->num_vectors.fetch_add(1, std::memory_order_relaxed) + 1;
    meta_guard->dirty = true;
    
    // Check if retraining is needed
    u64 last_train = meta_guard->last_train_count.load(std::memory_order_relaxed);
    if (retrain_interval_ > 0 && (new_count - last_train) >= retrain_interval_) {
        // TODO: Trigger online retraining
        // For now, just update the counter
        meta_guard->last_train_count.store(new_count, std::memory_order_relaxed);
    }
}

// --- Add Points Batch (SIMD optimized) ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::addPoints(const float* vectors, const u32* ids, u64 count, size_t num_threads) {
    if (!is_trained_.load(std::memory_order_acquire)) {
        throw std::runtime_error("IVFPQ: Index must be trained before adding vectors");
    }
    
    if (count == 0) return;
    
    // For small batches, use simple per-point insertion
    if (count < 32) {
        for (u64 i = 0; i < count; ++i) {
            addPoint(vectors + i * dim_, ids[i]);
        }
        return;
    }
    
    // =========================================================================
    // Phase 1: Batch find nearest centroids (SIMD optimized, parallel)
    // =========================================================================
    std::vector<u32> cluster_ids(count);
    findNearestCentroidsBatch(vectors, count, cluster_ids.data());
    
    // =========================================================================
    // Phase 2: Batch compute residuals (SIMD optimized, parallel)
    // =========================================================================
    std::vector<float> residuals(count * dim_);
    computeResidualsBatch(vectors, cluster_ids.data(), count, residuals.data());
    
    // =========================================================================
    // Phase 3: Batch encode residuals with PQ (SIMD optimized, parallel)
    // =========================================================================
    std::vector<u8> all_pq_codes(count * num_subquantizers_);
    encodeVectorsBatch(residuals.data(), count, all_pq_codes.data());
    
    // =========================================================================
    // Phase 4: Group vectors by cluster for efficient batch insertion
    // =========================================================================
    // Create cluster -> indices mapping for batch insert
    std::vector<std::vector<u64>> cluster_to_indices(num_clusters_);
    for (u64 i = 0; i < count; ++i) {
        cluster_to_indices[cluster_ids[i]].push_back(i);
    }
    
    // =========================================================================
    // Phase 5: Insert into inverted lists (can be parallelized per cluster)
    // =========================================================================
    if (num_threads <= 1) {
        // Single-threaded: insert directly
        for (u32 c = 0; c < num_clusters_; ++c) {
            const auto& indices = cluster_to_indices[c];
            for (u64 idx : indices) {
                appendToInvList(c, ids[idx], all_pq_codes.data() + idx * num_subquantizers_);
            }
        }
    } else {
        // Multi-threaded: process clusters in parallel
        // Note: different clusters have independent inverted lists, so no lock contention
        ThreadPool* pool = getOrCreateAddPool(num_threads);
        std::vector<std::future<void>> futures;
        
        for (u32 c = 0; c < num_clusters_; ++c) {
            if (cluster_to_indices[c].empty()) continue;
            
            futures.push_back(pool->enqueue([this, c, &cluster_to_indices, ids, &all_pq_codes]() {
                const auto& indices = cluster_to_indices[c];
                for (u64 idx : indices) {
                    appendToInvList(c, ids[idx], all_pq_codes.data() + idx * num_subquantizers_);
                }
            }));
        }
        
        for (auto& f : futures) {
            f.get();
        }
    }
    
    // =========================================================================
    // Phase 6: Update metadata count
    // =========================================================================
    GuardX<IVFPQMetadataPage> meta_guard(metadata_pid_);
    u64 new_count = meta_guard->num_vectors.fetch_add(count, std::memory_order_relaxed) + count;
    meta_guard->dirty = true;
    
    // Check if retraining is needed
    u64 last_train = meta_guard->last_train_count.load(std::memory_order_relaxed);
    if (retrain_interval_ > 0 && (new_count - last_train) >= retrain_interval_) {
        meta_guard->last_train_count.store(new_count, std::memory_order_relaxed);
    }
}

// --- Search ---
template <typename DistanceMetric>
template <bool stats>
std::vector<std::pair<float, u32>> IVFPQ<DistanceMetric>::search(
    const float* query, size_t k, size_t nprobe) {
    
    if (!is_trained_.load(std::memory_order_acquire)) {
        throw std::runtime_error("IVFPQ: Index must be trained before searching");
    }
    
    // Find nprobe nearest clusters
    auto nearest_clusters = findNearestCentroids(query, nprobe);
    
    if constexpr (stats) {
        stats_.lists_probed.fetch_add(nearest_clusters.size(), std::memory_order_relaxed);
    }
    
    // Max-heap for top-k results
    std::priority_queue<std::pair<float, u32>> result_heap;
    float heap_threshold = std::numeric_limits<float>::max();
    
    // Allocate contiguous buffer for batch-copying PQ codes (eliminates Guard overhead)
    std::vector<u8> code_buffer;
    std::vector<u32> id_buffer;
    
    // Precompute distance table storage (reused for each cluster)
    std::vector<float> flat_distance_table(num_subquantizers_ * IVFPQ_NUM_CODES);
    std::vector<float> query_residual(dim_);
    
    // Scan each cluster's inverted list
    for (const auto& [centroid_dist, cluster_id] : nearest_clusters) {
        // Get cache lock once per cluster
        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex_);
            
            if (!cache_valid_) {
                lock.unlock();
                loadCachesFromDisk();
                lock.lock();
            }
            
            // Compute query residual (query - centroid) for this cluster
            const float* centroid = centroids_cache_.data() + cluster_id * dim_;
            for (u32 d = 0; d < dim_; ++d) {
                query_residual[d] = query[d] - centroid[d];
            }
            
            // Compute distance table using the query residual
            // Use FAISS-style optimization: dis = ||cb||^2 - 2*<subquery, cb> + ||subquery||^2
            // The ||subquery||^2 term is constant and can be added once per subquantizer
            for (u32 m = 0; m < num_subquantizers_; ++m) {
                const float* subquery = query_residual.data() + m * subvector_dim_;
                const float* codebook = codebook_cache_[m].data();
                float* table_out = flat_distance_table.data() + m * IVFPQ_NUM_CODES;
                
#ifdef __AVX2__
                // Use precomputed codebook norms
                const float* cb_norms = codebook_norms_cache_[m].data();
                
                // Specialized path for subvector_dim=4 (common case: M=32, dim=128)
                if (subvector_dim_ == 4) {
                    // Load subquery once into SSE register (4 floats exactly)
                    __m128 sq = _mm_loadu_ps(subquery);
                    
                    // Compute ||subquery||^2 using SSE
                    __m128 sq_sq = _mm_mul_ps(sq, sq);
                    sq_sq = _mm_hadd_ps(sq_sq, sq_sq);
                    sq_sq = _mm_hadd_ps(sq_sq, sq_sq);
                    float subquery_norm = _mm_cvtss_f32(sq_sq);
                    
                    // Process 8 codebook entries at a time
                    u32 c = 0;
                    for (; c + 8 <= IVFPQ_NUM_CODES; c += 8) {
                        const float* cb = codebook + c * 4;
                        
                        // Load 8 codebook vectors (each 4 floats)
                        __m128 cb0 = _mm_loadu_ps(cb);
                        __m128 cb1 = _mm_loadu_ps(cb + 4);
                        __m128 cb2 = _mm_loadu_ps(cb + 8);
                        __m128 cb3 = _mm_loadu_ps(cb + 12);
                        __m128 cb4 = _mm_loadu_ps(cb + 16);
                        __m128 cb5 = _mm_loadu_ps(cb + 20);
                        __m128 cb6 = _mm_loadu_ps(cb + 24);
                        __m128 cb7 = _mm_loadu_ps(cb + 28);
                        
                        // Compute dot products: sq · cb
                        __m128 dp0 = _mm_mul_ps(sq, cb0);
                        __m128 dp1 = _mm_mul_ps(sq, cb1);
                        __m128 dp2 = _mm_mul_ps(sq, cb2);
                        __m128 dp3 = _mm_mul_ps(sq, cb3);
                        __m128 dp4 = _mm_mul_ps(sq, cb4);
                        __m128 dp5 = _mm_mul_ps(sq, cb5);
                        __m128 dp6 = _mm_mul_ps(sq, cb6);
                        __m128 dp7 = _mm_mul_ps(sq, cb7);
                        
                        // Horizontal sum each dot product
                        dp0 = _mm_hadd_ps(dp0, dp1);  // [dp0, dp0, dp1, dp1]
                        dp2 = _mm_hadd_ps(dp2, dp3);  // [dp2, dp2, dp3, dp3]
                        dp0 = _mm_hadd_ps(dp0, dp2);  // [dp0, dp1, dp2, dp3]
                        
                        dp4 = _mm_hadd_ps(dp4, dp5);
                        dp6 = _mm_hadd_ps(dp6, dp7);
                        dp4 = _mm_hadd_ps(dp4, dp6);  // [dp4, dp5, dp6, dp7]
                        
                        // Load norms
                        __m128 norms_lo = _mm_loadu_ps(cb_norms + c);
                        __m128 norms_hi = _mm_loadu_ps(cb_norms + c + 4);
                        
                        // dis = ||cb||^2 - 2*<subquery, cb> + ||subquery||^2
                        __m128 two = _mm_set1_ps(2.0f);
                        __m128 sq_norm = _mm_set1_ps(subquery_norm);
                        
                        __m128 dis_lo = _mm_sub_ps(norms_lo, _mm_mul_ps(two, dp0));
                        dis_lo = _mm_add_ps(dis_lo, sq_norm);
                        
                        __m128 dis_hi = _mm_sub_ps(norms_hi, _mm_mul_ps(two, dp4));
                        dis_hi = _mm_add_ps(dis_hi, sq_norm);
                        
                        // Store results
                        _mm_storeu_ps(table_out + c, dis_lo);
                        _mm_storeu_ps(table_out + c + 4, dis_hi);
                    }
                    
                    // Handle remaining (should be 0 for 256)
                    for (; c < IVFPQ_NUM_CODES; ++c) {
                        const float* cb = codebook + c * 4;
                        float dp = subquery[0]*cb[0] + subquery[1]*cb[1] + subquery[2]*cb[2] + subquery[3]*cb[3];
                        table_out[c] = cb_norms[c] - 2.0f * dp + subquery_norm;
                    }
                } else {
                    // General case for other subvector dimensions
                    // Compute ||subquery||^2 once
                    float subquery_norm = 0.0f;
                    for (u32 d = 0; d < subvector_dim_; ++d) {
                        subquery_norm += subquery[d] * subquery[d];
                    }
                    
                    // Process 4 codebook entries at a time
                    u32 c = 0;
                    for (; c + 4 <= IVFPQ_NUM_CODES; c += 4) {
                        const float* cb0 = codebook + c * subvector_dim_;
                        const float* cb1 = cb0 + subvector_dim_;
                        const float* cb2 = cb1 + subvector_dim_;
                        const float* cb3 = cb2 + subvector_dim_;
                        
                        float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f, dp3 = 0.0f;
                        for (u32 d = 0; d < subvector_dim_; ++d) {
                            float sq = subquery[d];
                            dp0 += sq * cb0[d];
                            dp1 += sq * cb1[d];
                            dp2 += sq * cb2[d];
                            dp3 += sq * cb3[d];
                        }
                        
                        table_out[c]     = cb_norms[c]     - 2.0f * dp0 + subquery_norm;
                        table_out[c + 1] = cb_norms[c + 1] - 2.0f * dp1 + subquery_norm;
                        table_out[c + 2] = cb_norms[c + 2] - 2.0f * dp2 + subquery_norm;
                        table_out[c + 3] = cb_norms[c + 3] - 2.0f * dp3 + subquery_norm;
                    }
                    
                    // Handle remaining codebook entries
                    for (; c < IVFPQ_NUM_CODES; ++c) {
                        const float* cb = codebook + c * subvector_dim_;
                        float dp = 0.0f;
                        for (u32 d = 0; d < subvector_dim_; ++d) {
                            dp += subquery[d] * cb[d];
                        }
                        table_out[c] = cb_norms[c] - 2.0f * dp + subquery_norm;
                    }
                }
#else
                // Fallback: naive distance computation
                for (u32 c = 0; c < IVFPQ_NUM_CODES; ++c) {
                    table_out[c] = computeSubvectorDistance(subquery, codebook + c * subvector_dim_, subvector_dim_);
                }
#endif
            }
        }
        
        PID dir_page_pid = getClusterDirPage(cluster_id);
        u32 local_idx = getClusterLocalIdx(cluster_id);
        
        // Read directory entry (relaxed read is OK for search)
        GuardORelaxed<InvListDirPage> dir_guard(dir_page_pid);
        const InvListEntry* entry = dir_guard->getEntry(local_idx);
        
        u32 list_size = entry->list_size.load(std::memory_order_acquire);
        if (list_size == 0) continue;
        
        // Prefetch cached pages using BufferManager interface
        u32 cached_count = entry->cached_page_count.load(std::memory_order_acquire);
        if (cached_count > 0) {
            // Collect valid PIDs and offsets for batch prefetch
            constexpr u32 MAX_PREFETCH = IVFPQ_MAX_CACHED_PAGES;
            PID prefetch_pids[MAX_PREFETCH];
            u32 prefetch_offsets[MAX_PREFETCH];
            u32 prefetch_count = 0;
            
            for (u32 i = 0; i < cached_count && i < IVFPQ_MAX_CACHED_PAGES; ++i) {
                if (entry->cached_pages[i] != BufferManager::invalidPID) {
                    prefetch_pids[prefetch_count] = entry->cached_pages[i];
                    // Prefetch at the start of entry data (first PQ code entry)
                    prefetch_offsets[prefetch_count] = 0;
                    prefetch_count++;
                }
            }
            
            if (prefetch_count > 0) {
                bm.prefetchPages(prefetch_pids, prefetch_count, prefetch_offsets);
            }
        }
        
        // Traverse inverted list
        PID current_page = entry->first_page_pid;
        u32 vectors_scanned = 0;
        u32 pages_visited = 0;
        const u32 MAX_PAGES = 10000;  // Safety limit
        
        while (current_page != BufferManager::invalidPID) {
            if (++pages_visited > MAX_PAGES) {
                // Prevent infinite loop from corrupted page chain
                CALIBY_LOG_WARN("IVFPQ", "Exceeded maximum pages visited in inverted list traversal.");
                break;
            }
            
            GuardORelaxed<InvListDataPage> page_guard(current_page);
            const InvListDataPage* page_ptr = page_guard.ptr;  // Cache page pointer
            u32 count = page_ptr->count;
            
            // Bounds check
            if (count > entries_per_invlist_page_) {
                // Corrupted count, skip this page
                break;
            }
            
            // // Prefetch next page
            PID next_page = page_ptr->next_page;
            // if (next_page != BufferManager::invalidPID) {
            //     _mm_prefetch(reinterpret_cast<const char*>(&bm.getPageState(next_page)), _MM_HINT_T0);
            // }
            
            // SIMD-optimized distance computation
            const float* dis_table = flat_distance_table.data();
            
#ifdef __AVX2__
            {
                const u8* entry_base = page_ptr->getEntryData();
                
                // Prefetch next page data for better memory access patterns
                if (next_page != BufferManager::invalidPID) {
                    _mm_prefetch(reinterpret_cast<const char*>(&bm.getPageState(next_page)), _MM_HINT_T0);
                }
                
                // Process 4 codes at a time using SIMD
                u32 i = 0;
                const u32 count4 = (count / 4) * 4;
                
                // Use specialized M=32 function for common SIFT case
                const bool use_m32 = (num_subquantizers_ == 32);
                
                for (; i < count4; i += 4) {
                    // Prefetch codes for next iteration
                    if (i + 4 < count4) {
                        _mm_prefetch(reinterpret_cast<const char*>(entry_base + (i + 4) * pq_entry_size_), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(entry_base + (i + 5) * pq_entry_size_), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(entry_base + (i + 6) * pq_entry_size_), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(entry_base + (i + 7) * pq_entry_size_), _MM_HINT_T0);
                    }
                    
                    // Direct pointer arithmetic
                    const PQCodeEntry* e0 = reinterpret_cast<const PQCodeEntry*>(entry_base + i * pq_entry_size_);
                    const PQCodeEntry* e1 = reinterpret_cast<const PQCodeEntry*>(entry_base + (i + 1) * pq_entry_size_);
                    const PQCodeEntry* e2 = reinterpret_cast<const PQCodeEntry*>(entry_base + (i + 2) * pq_entry_size_);
                    const PQCodeEntry* e3 = reinterpret_cast<const PQCodeEntry*>(entry_base + (i + 3) * pq_entry_size_);
                    
                    float dist0, dist1, dist2, dist3;
                    
                    if (use_m32) {
                        // Use specialized M=32 function
                        distance_four_codes_m32(dis_table,
                            e0->getCodes(), e1->getCodes(), e2->getCodes(), e3->getCodes(),
                            dist0, dist1, dist2, dist3);
                    } else {
                        distance_four_codes_general(dis_table,
                            e0->getCodes(), e1->getCodes(), e2->getCodes(), e3->getCodes(),
                            num_subquantizers_,
                            dist0, dist1, dist2, dist3);
                    }
                    
                    // Optimized heap updates with threshold (FAISS-style batched check)
                    // Check all 4 distances against threshold first
                    u32 id0 = e0->original_id;
                    u32 id1 = e1->original_id;
                    u32 id2 = e2->original_id;
                    u32 id3 = e3->original_id;
                    
                    if (result_heap.size() < k) {
                        result_heap.push({dist0, id0});
                        if (result_heap.size() == k) heap_threshold = result_heap.top().first;
                    } else if (dist0 < heap_threshold) {
                        result_heap.pop();
                        result_heap.push({dist0, id0});
                        heap_threshold = result_heap.top().first;
                    }
                    
                    if (result_heap.size() < k) {
                        result_heap.push({dist1, id1});
                        if (result_heap.size() == k) heap_threshold = result_heap.top().first;
                    } else if (dist1 < heap_threshold) {
                        result_heap.pop();
                        result_heap.push({dist1, id1});
                        heap_threshold = result_heap.top().first;
                    }
                    
                    if (result_heap.size() < k) {
                        result_heap.push({dist2, id2});
                        if (result_heap.size() == k) heap_threshold = result_heap.top().first;
                    } else if (dist2 < heap_threshold) {
                        result_heap.pop();
                        result_heap.push({dist2, id2});
                        heap_threshold = result_heap.top().first;
                    }
                    
                    if (result_heap.size() < k) {
                        result_heap.push({dist3, id3});
                        if (result_heap.size() == k) heap_threshold = result_heap.top().first;
                    } else if (dist3 < heap_threshold) {
                        result_heap.pop();
                        result_heap.push({dist3, id3});
                        heap_threshold = result_heap.top().first;
                    }
                }
                
                // Handle remaining codes (less than 4) with generalized SIMD single
                for (; i < count; ++i) {
                    const PQCodeEntry* pq_entry = reinterpret_cast<const PQCodeEntry*>(entry_base + i * pq_entry_size_);
                    float dist = distance_single_code_general(dis_table, pq_entry->getCodes(), num_subquantizers_);
                    
                    if (result_heap.size() < k) {
                        result_heap.push({dist, pq_entry->original_id});
                        if (result_heap.size() == k) heap_threshold = result_heap.top().first;
                    } else if (dist < heap_threshold) {
                        result_heap.pop();
                        result_heap.push({dist, pq_entry->original_id});
                        heap_threshold = result_heap.top().first;
                    }
                }
                
                vectors_scanned += count;
            }
#else
            {
                // Fallback: scalar loop (no AVX2)
                for (u32 i = 0; i < count; ++i) {
                    const PQCodeEntry* pq_entry = page_ptr->getEntry(i, pq_entry_size_);
                    const u8* codes = pq_entry->getCodes();
                    
                    float dist = 0.0f;
                    const float* table_ptr = dis_table;
                    for (u32 m = 0; m < num_subquantizers_; ++m) {
                        dist += table_ptr[codes[m]];
                        table_ptr += IVFPQ_NUM_CODES;
                    }
                    
                    if (result_heap.size() < k) {
                        result_heap.push({dist, pq_entry->original_id});
                        if (result_heap.size() == k) heap_threshold = result_heap.top().first;
                    } else if (dist < heap_threshold) {
                        result_heap.pop();
                        result_heap.push({dist, pq_entry->original_id});
                        heap_threshold = result_heap.top().first;
                    }
                    
                    vectors_scanned++;
                }
            }
#endif
            
            current_page = next_page;
        }
        
        if constexpr (stats) {
            stats_.vectors_scanned.fetch_add(vectors_scanned, std::memory_order_relaxed);
            stats_.dist_comps.fetch_add(vectors_scanned, std::memory_order_relaxed);
        }
    }
    
    // Convert heap to sorted result
    std::vector<std::pair<float, u32>> results;
    results.reserve(result_heap.size());
    while (!result_heap.empty()) {
        results.push_back(result_heap.top());
        result_heap.pop();
    }
    std::reverse(results.begin(), results.end());
    
    return results;
}

// --- Search Batch ---
template <typename DistanceMetric>
template <bool stats>
std::vector<std::vector<std::pair<float, u32>>> IVFPQ<DistanceMetric>::searchBatch(
    std::span<const float> queries, size_t k, size_t nprobe, size_t num_threads) {
    
    size_t num_queries = queries.size() / dim_;
    std::vector<std::vector<std::pair<float, u32>>> results(num_queries);
    
    if (num_threads <= 1) {
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = search<stats>(queries.data() + i * dim_, k, nprobe);
        }
    } else {
        ThreadPool* pool = getOrCreateSearchPool(num_threads);
        std::vector<std::future<std::vector<std::pair<float, u32>>>> futures;
        futures.reserve(num_queries);
        
        for (size_t i = 0; i < num_queries; ++i) {
            const float* q = queries.data() + i * dim_;
            futures.push_back(pool->enqueue([this, q, k, nprobe]() {
                return search<stats>(q, k, nprobe);
            }));
        }
        
        for (size_t i = 0; i < num_queries; ++i) {
            results[i] = futures[i].get();
        }
    }
    
    return results;
}

// --- Size ---
template <typename DistanceMetric>
u64 IVFPQ<DistanceMetric>::size() const {
    GuardO<IVFPQMetadataPage> meta_guard(metadata_pid_);
    return meta_guard->num_vectors.load(std::memory_order_acquire);
}

// --- Get Stats ---
template <typename DistanceMetric>
IVFPQStats IVFPQ<DistanceMetric>::getStats() const {
    IVFPQStats result = stats_;
    result.num_clusters = num_clusters_;
    result.num_subquantizers = num_subquantizers_;
    
    // Compute list sizes
    result.list_sizes.resize(num_clusters_);
    u64 total_size = 0;
    
    for (u32 c = 0; c < num_clusters_; ++c) {
        PID dir_page_pid = getClusterDirPage(c);
        u32 local_idx = getClusterLocalIdx(c);
        
        try {
            GuardO<InvListDirPage> dir_guard(dir_page_pid);
            const InvListEntry* entry = dir_guard->getEntry(local_idx);
            result.list_sizes[c] = entry->list_size.load(std::memory_order_relaxed);
            total_size += result.list_sizes[c];
        } catch (const OLCRestartException&) {
            result.list_sizes[c] = 0;
        }
    }
    
    result.avg_list_size = num_clusters_ > 0 ? static_cast<double>(total_size) / num_clusters_ : 0.0;
    
    return result;
}

// --- Flush ---
template <typename DistanceMetric>
void IVFPQ<DistanceMetric>::flush() {
    bm.flushAll();
}

// --- Explicit Template Instantiations ---
template class IVFPQ<L2Distance>;

// Explicit instantiation of member function templates for search
template std::vector<std::pair<float, u32>> IVFPQ<L2Distance>::search<false>(const float*, size_t, size_t);
template std::vector<std::pair<float, u32>> IVFPQ<L2Distance>::search<true>(const float*, size_t, size_t);

template std::vector<std::vector<std::pair<float, u32>>> IVFPQ<L2Distance>::searchBatch<false>(std::span<const float>, size_t, size_t, size_t);
template std::vector<std::vector<std::pair<float, u32>>> IVFPQ<L2Distance>::searchBatch<true>(std::span<const float>, size_t, size_t, size_t);

// Note: InnerProductDistance is currently aliased to L2Distance
// Once a proper inner product implementation exists, uncomment below:
// template class IVFPQ<InnerProductDistance>;
