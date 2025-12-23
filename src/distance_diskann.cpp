// TODO
// CHECK COSINE ON LINUX

#ifdef _WINDOWS
#include <immintrin.h>
#include <intrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#else
#include <immintrin.h>
#endif

#include <iostream>
#include <sstream>

#include "distance_diskann.hpp"
#include "simd_utils.hpp"


//
// Base Class Implementatons
//
float Distance::compare(const float *a, const float *b, const float normA, const float normB, uint32_t length) const {
    throw std::logic_error("This function is not implemented.");
}


uint32_t Distance::post_normalization_dimension(uint32_t orig_dimension) const {
    return orig_dimension;
}

Metric Distance::get_metric() const {
    return _distance_metric;
}

bool Distance::preprocessing_required() const {
    return false;
}

void Distance::preprocess_base_points(float *original_data, const size_t orig_dim, const size_t num_points) {}

void Distance::preprocess_query(const float *query_vec, const size_t query_dim, float *scratch_query) {
    std::memcpy(scratch_query, query_vec, query_dim * sizeof(float));
}

size_t Distance::get_required_alignment() const {
    return _alignment_factor;
}

#ifndef _WINDOWS
float Distance::compare(const float *a, const float *b, uint32_t size) const {
    a = (const float *)__builtin_assume_aligned(a, 32);
    b = (const float *)__builtin_assume_aligned(b, 32);
#else
float Distance::compare(const float *a, const float *b, uint32_t size) const {
#endif

    float result = 0;
#ifdef USE_AVX2
    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(size / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[8j:8j+7], b[8j:8j+7]
        // load a_vec
        if (j < (niters - 1)) {
            _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_load_ps(a + 8 * j);
        // load b_vec
        __m256 b_vec = _mm256_load_ps(b + 8 * j);
        // a_vec - b_vec
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);

        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }

    // horizontal add sum
    result = _mm256_reduce_add_ps(sum);
#else
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 32)
#endif
    for (int32_t i = 0; i < (int32_t)size; i++) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
    }
#endif
    return result;
}
