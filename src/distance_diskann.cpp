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
#include "distance.hpp"
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

// Use the optimized SIMD implementation from distance.hpp
float Distance::compare(const float *a, const float *b, uint32_t size) const {
    return hnsw_distance::SIMDAcceleratedL2::compare(a, b, size);
}
