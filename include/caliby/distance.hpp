#pragma once

#include <cstddef>

// Include the header for all SIMD intrinsics.
// For MSVC, you might need <intrin.h>. For GCC/Clang, <immintrin.h> is sufficient.
#include <immintrin.h>

namespace hnsw_distance {

// --- Scalar Fallback Implementation ---
// This version is used when no SIMD instructions are available.
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
    alignas(32) float TmpRes[8];
    size_t qty16 = qty >> 4;  // qty / 16
    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        // First 8 elements
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        // Second 8 elements
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
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

} // namespace hnsw_distance