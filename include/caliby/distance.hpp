#pragma once

#include <cstddef>
#include <cmath>

// Include the header for all SIMD intrinsics.
// For MSVC, you might need <intrin.h>. For GCC/Clang, <immintrin.h> is sufficient.
#include <immintrin.h>

namespace hnsw_distance {

//=============================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
//=============================================================================

// --- L2 Squared Distance (Scalar) ---
static inline float
L2SqrScalar(const float *pVect1, const float *pVect2, size_t qty) {
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return res;
}

// --- Inner Product (Scalar) ---
static inline float
InnerProductScalar(const float *pVect1, const float *pVect2, size_t qty) {
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += (*pVect1) * (*pVect2);
        pVect1++;
        pVect2++;
    }
    return res;
}

// --- Negative Inner Product for distance (Scalar) ---
// Returns 1 - IP to convert similarity to distance (lower is better)
static inline float
NegativeInnerProductScalar(const float *pVect1, const float *pVect2, size_t qty) {
    return 1.0f - InnerProductScalar(pVect1, pVect2, qty);
}

#if defined(__AVX512F__)
// --- AVX-512 Implementation (processes 16 elements at once) ---
static inline float
L2Sqr_AVX512_16(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(64) float TmpRes[16];
    size_t qty16 = qty >> 4;  // qty / 16
    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
               TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
               TmpRes[13] + TmpRes[14] + TmpRes[15];
    return res;
}

static inline float
L2Sqr_AVX512_16_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4 << 4;  // Align to 16
    float res = L2Sqr_AVX512_16(pVect1, pVect2, qty16);
    
    size_t qty_left = qty - qty16;
    if (qty_left > 0) {
        res += L2SqrScalar(pVect1 + qty16, pVect2 + qty16, qty_left);
    }
    return res;
}
#endif

#if defined(__AVX__)
// --- AVX Implementation (processes 16 elements using two 8-element chunks) ---
static inline float
L2Sqr_AVX_16(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty32 = qty >> 5;
    const float *pEnd32 = pVect1 + (qty32 << 5);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    while (pVect1 < pEnd32) {
        __m256 v1_0 = _mm256_loadu_ps(pVect1);
        __m256 v2_0 = _mm256_loadu_ps(pVect2);
        __m256 v1_1 = _mm256_loadu_ps(pVect1 + 8);
        __m256 v2_1 = _mm256_loadu_ps(pVect2 + 8);
        __m256 v1_2 = _mm256_loadu_ps(pVect1 + 16);
        __m256 v2_2 = _mm256_loadu_ps(pVect2 + 16);
        __m256 v1_3 = _mm256_loadu_ps(pVect1 + 24);
        __m256 v2_3 = _mm256_loadu_ps(pVect2 + 24);
        __m256 d0 = _mm256_sub_ps(v1_0, v2_0);
        __m256 d1 = _mm256_sub_ps(v1_1, v2_1);
        __m256 d2 = _mm256_sub_ps(v1_2, v2_2);
        __m256 d3 = _mm256_sub_ps(v1_3, v2_3);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
        pVect1 += 32;
        pVect2 += 32;
    }

    size_t remaining = (qty >> 4) & 1;
    if (remaining) {
        __m256 v1_0 = _mm256_loadu_ps(pVect1);
        __m256 v2_0 = _mm256_loadu_ps(pVect2);
        __m256 v1_1 = _mm256_loadu_ps(pVect1 + 8);
        __m256 v2_1 = _mm256_loadu_ps(pVect2 + 8);
        __m256 d0 = _mm256_sub_ps(v1_0, v2_0);
        __m256 d1 = _mm256_sub_ps(v1_1, v2_1);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);
        pVect1 += 16;
        pVect2 += 16;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    __m128 vlow = _mm256_castps256_ps128(sum0);
    __m128 vhigh = _mm256_extractf128_ps(sum0, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float
L2Sqr_AVX_16_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4 << 4;  // Align to 16
    float res = L2Sqr_AVX_16(pVect1, pVect2, qty16);
    
    size_t qty_left = qty - qty16;
    if (qty_left > 0) {
        res += L2SqrScalar(pVect1 + qty16, pVect2 + qty16, qty_left);
    }
    return res;
}
#endif

#if defined(__SSE__)
// --- SSE Implementation (processes 16 elements using four 4-element chunks) ---
static inline float
L2Sqr_SSE_16(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(32) float TmpRes[8];
    size_t qty16 = qty >> 4;  // qty / 16
    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        // First 4 elements
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        // Second 4 elements
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        // Third 4 elements
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        // Fourth 4 elements
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

// --- SSE Implementation for 4-element chunks ---
static inline float
L2Sqr_SSE_4(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(32) float TmpRes[8];
    size_t qty4 = qty >> 2;  // qty / 4
    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static inline float
L2Sqr_SSE_16_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4 << 4;  // Align to 16
    float res = L2Sqr_SSE_16(pVect1, pVect2, qty16);
    
    size_t qty_left = qty - qty16;
    if (qty_left > 0) {
        res += L2SqrScalar(pVect1 + qty16, pVect2 + qty16, qty_left);
    }
    return res;
}

static inline float
L2Sqr_SSE_4_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty4 = qty >> 2 << 2;  // Align to 4
    float res = L2Sqr_SSE_4(pVect1, pVect2, qty4);
    
    size_t qty_left = qty - qty4;
    if (qty_left > 0) {
        res += L2SqrScalar(pVect1 + qty4, pVect2 + qty4, qty_left);
    }
    return res;
}
#endif

//=============================================================================
// INNER PRODUCT SIMD IMPLEMENTATIONS
//=============================================================================

#if defined(__AVX512F__)
// --- AVX-512 Inner Product Implementation (16 elements at once) ---
static inline float
InnerProduct_AVX512_16(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4;
    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 sum = _mm512_setzero_ps();

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        sum = _mm512_fmadd_ps(v1, v2, sum);
        pVect1 += 16;
        pVect2 += 16;
    }

    return _mm512_reduce_add_ps(sum);
}

static inline float
InnerProduct_AVX512_16_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProduct_AVX512_16(pVect1, pVect2, qty16);
    
    size_t qty_left = qty - qty16;
    if (qty_left > 0) {
        res += InnerProductScalar(pVect1 + qty16, pVect2 + qty16, qty_left);
    }
    return res;
}
#endif

#if defined(__AVX__)
// --- AVX Inner Product Implementation (32 elements unrolled) ---
static inline float
InnerProduct_AVX_16(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty32 = qty >> 5;
    const float *pEnd32 = pVect1 + (qty32 << 5);
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    while (pVect1 < pEnd32) {
        __m256 v1_0 = _mm256_loadu_ps(pVect1);
        __m256 v2_0 = _mm256_loadu_ps(pVect2);
        __m256 v1_1 = _mm256_loadu_ps(pVect1 + 8);
        __m256 v2_1 = _mm256_loadu_ps(pVect2 + 8);
        __m256 v1_2 = _mm256_loadu_ps(pVect1 + 16);
        __m256 v2_2 = _mm256_loadu_ps(pVect2 + 16);
        __m256 v1_3 = _mm256_loadu_ps(pVect1 + 24);
        __m256 v2_3 = _mm256_loadu_ps(pVect2 + 24);
        sum0 = _mm256_fmadd_ps(v1_0, v2_0, sum0);
        sum1 = _mm256_fmadd_ps(v1_1, v2_1, sum1);
        sum2 = _mm256_fmadd_ps(v1_2, v2_2, sum2);
        sum3 = _mm256_fmadd_ps(v1_3, v2_3, sum3);
        pVect1 += 32;
        pVect2 += 32;
    }

    size_t remaining = (qty >> 4) & 1;
    if (remaining) {
        __m256 v1_0 = _mm256_loadu_ps(pVect1);
        __m256 v2_0 = _mm256_loadu_ps(pVect2);
        __m256 v1_1 = _mm256_loadu_ps(pVect1 + 8);
        __m256 v2_1 = _mm256_loadu_ps(pVect2 + 8);
        sum0 = _mm256_fmadd_ps(v1_0, v2_0, sum0);
        sum1 = _mm256_fmadd_ps(v1_1, v2_1, sum1);
        pVect1 += 16;
        pVect2 += 16;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    __m128 vlow = _mm256_castps256_ps128(sum0);
    __m128 vhigh = _mm256_extractf128_ps(sum0, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float
InnerProduct_AVX_16_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProduct_AVX_16(pVect1, pVect2, qty16);
    
    size_t qty_left = qty - qty16;
    if (qty_left > 0) {
        res += InnerProductScalar(pVect1 + qty16, pVect2 + qty16, qty_left);
    }
    return res;
}
#endif

#if defined(__SSE__)
// --- SSE Inner Product Implementation (16 elements per iteration) ---
static inline float
InnerProduct_SSE_16(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(32) float TmpRes[4];
    size_t qty16 = qty >> 4;
    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 sum = _mm_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(pVect1);
        __m128 v2 = _mm_loadu_ps(pVect2);
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        pVect1 += 4;
        pVect2 += 4;

        v1 = _mm_loadu_ps(pVect1);
        v2 = _mm_loadu_ps(pVect2);
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        pVect1 += 4;
        pVect2 += 4;

        v1 = _mm_loadu_ps(pVect1);
        v2 = _mm_loadu_ps(pVect2);
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        pVect1 += 4;
        pVect2 += 4;

        v1 = _mm_loadu_ps(pVect1);
        v2 = _mm_loadu_ps(pVect2);
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static inline float
InnerProduct_SSE_4(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(32) float TmpRes[4];
    size_t qty4 = qty >> 2;
    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 sum = _mm_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(pVect1);
        __m128 v2 = _mm_loadu_ps(pVect2);
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static inline float
InnerProduct_SSE_16_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProduct_SSE_16(pVect1, pVect2, qty16);
    
    size_t qty_left = qty - qty16;
    if (qty_left > 0) {
        res += InnerProductScalar(pVect1 + qty16, pVect2 + qty16, qty_left);
    }
    return res;
}

static inline float
InnerProduct_SSE_4_Residuals(const float *pVect1, const float *pVect2, size_t qty) {
    size_t qty4 = qty >> 2 << 2;
    float res = InnerProduct_SSE_4(pVect1, pVect2, qty4);
    
    size_t qty_left = qty - qty4;
    if (qty_left > 0) {
        res += InnerProductScalar(pVect1 + qty4, pVect2 + qty4, qty_left);
    }
    return res;
}
#endif

//=============================================================================
// DISTANCE METRIC STRUCTS (Template Parameters for Index Classes)
//=============================================================================

// This is the main distance metric struct that will be used as a template parameter.
// It automatically selects the best available SIMD implementation following hnswlib's logic.
struct SIMDAcceleratedL2 {
    static float compare(const float* v1, const float* v2, size_t dim) {
    #if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
        // Choose the optimal function based on dimension alignment, like hnswlib
        if (dim % 16 == 0) {
            // Use 16-element processing for dimensions divisible by 16
            #if defined(__AVX512F__)
                return L2Sqr_AVX512_16(v1, v2, dim);
            #elif defined(__AVX__)
                return L2Sqr_AVX_16(v1, v2, dim);
            #elif defined(__SSE__)
                return L2Sqr_SSE_16(v1, v2, dim);
            #endif
        } else if (dim % 4 == 0) {
            // Use 4-element processing for dimensions divisible by 4
            #if defined(__SSE__)
                return L2Sqr_SSE_4(v1, v2, dim);
            #else
                return L2SqrScalar(v1, v2, dim);
            #endif
        } else if (dim > 16) {
            // Use 16-element processing with residuals for large dimensions
            #if defined(__AVX512F__)
                return L2Sqr_AVX512_16_Residuals(v1, v2, dim);
            #elif defined(__AVX__)
                return L2Sqr_AVX_16_Residuals(v1, v2, dim);
            #elif defined(__SSE__)
                return L2Sqr_SSE_16_Residuals(v1, v2, dim);
            #endif
        } else if (dim > 4) {
            // Use 4-element processing with residuals for medium dimensions
            #if defined(__SSE__)
                return L2Sqr_SSE_4_Residuals(v1, v2, dim);
            #else
                return L2SqrScalar(v1, v2, dim);
            #endif
        } else {
            // Fall back to scalar for small dimensions
            return L2SqrScalar(v1, v2, dim);
        }
    #else
        // No SIMD available, use scalar implementation
        return L2SqrScalar(v1, v2, dim);
    #endif
    }
};

/**
 * Inner Product Distance (negative inner product for distance ranking)
 * 
 * For similarity search, higher inner product means more similar.
 * To use as distance (lower = better), we return 1 - IP.
 * 
 * Note: For MIPS (Maximum Inner Product Search), vectors should be normalized
 * for this to be equivalent to cosine similarity.
 */
struct SIMDAcceleratedIP {
    static float compare(const float* v1, const float* v2, size_t dim) {
        float ip;
    #if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
        if (dim % 16 == 0) {
            #if defined(__AVX512F__)
                ip = InnerProduct_AVX512_16(v1, v2, dim);
            #elif defined(__AVX__)
                ip = InnerProduct_AVX_16(v1, v2, dim);
            #elif defined(__SSE__)
                ip = InnerProduct_SSE_16(v1, v2, dim);
            #endif
        } else if (dim % 4 == 0) {
            #if defined(__SSE__)
                ip = InnerProduct_SSE_4(v1, v2, dim);
            #else
                ip = InnerProductScalar(v1, v2, dim);
            #endif
        } else if (dim > 16) {
            #if defined(__AVX512F__)
                ip = InnerProduct_AVX512_16_Residuals(v1, v2, dim);
            #elif defined(__AVX__)
                ip = InnerProduct_AVX_16_Residuals(v1, v2, dim);
            #elif defined(__SSE__)
                ip = InnerProduct_SSE_16_Residuals(v1, v2, dim);
            #endif
        } else if (dim > 4) {
            #if defined(__SSE__)
                ip = InnerProduct_SSE_4_Residuals(v1, v2, dim);
            #else
                ip = InnerProductScalar(v1, v2, dim);
            #endif
        } else {
            ip = InnerProductScalar(v1, v2, dim);
        }
    #else
        ip = InnerProductScalar(v1, v2, dim);
    #endif
        // Return 1 - IP to convert to distance (lower = more similar)
        return 1.0f - ip;
    }
    
    // Get raw inner product (for cases where similarity is needed directly)
    static float inner_product(const float* v1, const float* v2, size_t dim) {
    #if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
        if (dim % 16 == 0) {
            #if defined(__AVX512F__)
                return InnerProduct_AVX512_16(v1, v2, dim);
            #elif defined(__AVX__)
                return InnerProduct_AVX_16(v1, v2, dim);
            #elif defined(__SSE__)
                return InnerProduct_SSE_16(v1, v2, dim);
            #endif
        } else if (dim % 4 == 0) {
            #if defined(__SSE__)
                return InnerProduct_SSE_4(v1, v2, dim);
            #else
                return InnerProductScalar(v1, v2, dim);
            #endif
        } else if (dim > 16) {
            #if defined(__AVX512F__)
                return InnerProduct_AVX512_16_Residuals(v1, v2, dim);
            #elif defined(__AVX__)
                return InnerProduct_AVX_16_Residuals(v1, v2, dim);
            #elif defined(__SSE__)
                return InnerProduct_SSE_16_Residuals(v1, v2, dim);
            #endif
        } else if (dim > 4) {
            #if defined(__SSE__)
                return InnerProduct_SSE_4_Residuals(v1, v2, dim);
            #else
                return InnerProductScalar(v1, v2, dim);
            #endif
        } else {
            return InnerProductScalar(v1, v2, dim);
        }
    #else
        return InnerProductScalar(v1, v2, dim);
    #endif
    }
};

/**
 * Cosine Distance
 * 
 * Computes 1 - cosine_similarity = 1 - (a·b)/(||a|| * ||b||)
 * 
 * Note: If vectors are pre-normalized, cosine distance equals 1 - inner_product.
 * For efficiency, prefer normalizing vectors at insertion time and using IP distance.
 */
struct SIMDAcceleratedCosine {
    static float compare(const float* v1, const float* v2, size_t dim) {
        // Compute inner product and norms simultaneously for efficiency
        float ip, norm1_sq, norm2_sq;
        
    #if defined(__AVX__)
        if (dim >= 16) {
            __m256 sum_ip = _mm256_setzero_ps();
            __m256 sum_n1 = _mm256_setzero_ps();
            __m256 sum_n2 = _mm256_setzero_ps();
            
            size_t qty8 = dim >> 3;
            const float* pEnd = v1 + (qty8 << 3);
            
            while (v1 < pEnd) {
                __m256 vec1 = _mm256_loadu_ps(v1);
                __m256 vec2 = _mm256_loadu_ps(v2);
                sum_ip = _mm256_fmadd_ps(vec1, vec2, sum_ip);
                sum_n1 = _mm256_fmadd_ps(vec1, vec1, sum_n1);
                sum_n2 = _mm256_fmadd_ps(vec2, vec2, sum_n2);
                v1 += 8;
                v2 += 8;
            }
            
            // Horizontal sum for AVX
            __m128 vlow_ip = _mm256_castps256_ps128(sum_ip);
            __m128 vhigh_ip = _mm256_extractf128_ps(sum_ip, 1);
            vlow_ip = _mm_add_ps(vlow_ip, vhigh_ip);
            __m128 shuf = _mm_movehdup_ps(vlow_ip);
            __m128 sums = _mm_add_ps(vlow_ip, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            ip = _mm_cvtss_f32(sums);
            
            __m128 vlow_n1 = _mm256_castps256_ps128(sum_n1);
            __m128 vhigh_n1 = _mm256_extractf128_ps(sum_n1, 1);
            vlow_n1 = _mm_add_ps(vlow_n1, vhigh_n1);
            shuf = _mm_movehdup_ps(vlow_n1);
            sums = _mm_add_ps(vlow_n1, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            norm1_sq = _mm_cvtss_f32(sums);
            
            __m128 vlow_n2 = _mm256_castps256_ps128(sum_n2);
            __m128 vhigh_n2 = _mm256_extractf128_ps(sum_n2, 1);
            vlow_n2 = _mm_add_ps(vlow_n2, vhigh_n2);
            shuf = _mm_movehdup_ps(vlow_n2);
            sums = _mm_add_ps(vlow_n2, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            norm2_sq = _mm_cvtss_f32(sums);
            
            // Handle residuals
            size_t remaining = dim - (qty8 << 3);
            const float* v1_start = v1;
            const float* v2_start = v2;
            for (size_t i = 0; i < remaining; i++) {
                ip += v1_start[i] * v2_start[i];
                norm1_sq += v1_start[i] * v1_start[i];
                norm2_sq += v2_start[i] * v2_start[i];
            }
        } else
    #endif
        {
            // Scalar fallback
            ip = 0.0f;
            norm1_sq = 0.0f;
            norm2_sq = 0.0f;
            for (size_t i = 0; i < dim; i++) {
                ip += v1[i] * v2[i];
                norm1_sq += v1[i] * v1[i];
                norm2_sq += v2[i] * v2[i];
            }
        }
        
        // Compute cosine distance: 1 - cosine_similarity
        float norm_product = std::sqrt(norm1_sq * norm2_sq);
        if (norm_product < 1e-10f) {
            return 1.0f;  // Return max distance for zero vectors
        }
        float cosine_sim = ip / norm_product;
        // Clamp to [-1, 1] to handle floating point errors
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));
        return 1.0f - cosine_sim;
    }
};

// Alias for backward compatibility
using L2Distance = SIMDAcceleratedL2;
using InnerProductDistance = SIMDAcceleratedIP;
using CosineDistance = SIMDAcceleratedCosine;

} // namespace hnsw_distance