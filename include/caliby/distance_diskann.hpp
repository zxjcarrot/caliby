#pragma once
#include <cstring>
#include "utils.hpp"

enum Metric
{
    L2 = 0,
    INNER_PRODUCT = 1,
    COSINE = 2,
    FAST_L2 = 3
};

class Distance
{
  public:
    Distance(Metric dist_metric) : _distance_metric(dist_metric)
    {
    }

    // distance comparison function
    float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));

    // Needed only for COSINE-BYTE and INNER_PRODUCT-BYTE
    float compare(const float *a, const float *b, const float normA, const float normB,
                                            uint32_t length) const;

    // For MIPS, normalization adds an extra dimension to the vectors.
    // This function lets callers know if the normalization process
    // changes the dimension.
    uint32_t post_normalization_dimension(uint32_t orig_dimension) const;

    Metric get_metric() const;

    // This is for efficiency. If no normalization is required, the callers
    // can simply ignore the normalize_data_for_build() function.
    bool preprocessing_required() const;

    // Check the preprocessing_required() function before calling this.
    // Clients can call the function like this:
    //
    //  if (metric->preprocessing_required()){
    //     float* normalized_data_batch;
    //      Split data into batches of batch_size and for each, call:
    //       metric->preprocess_base_points(data_batch, batch_size);
    //
    //  TODO: This does not take into account the case for SSD inner product
    //  where the dimensions change after normalization.
    void preprocess_base_points(float *original_data, const size_t orig_dim,
                                                          const size_t num_points);

    // Invokes normalization for a single vector during search. The scratch space
    // has to be created by the caller keeping track of the fact that
    // normalization might change the dimension of the query vector.
    void preprocess_query(const float *query_vec, const size_t query_dim, float *scratch_query);

    // If an algorithm has a requirement that some data be aligned to a certain
    // boundary it can use this function to indicate that requirement. Currently,
    // we are setting it to 8 because that works well for AVX2. If we have AVX512
    // implementations of distance algos, they might have to set this to 16
    // (depending on how they are implemented)
    size_t get_required_alignment() const;

    // Providing a default implementation for the destructor because we
    // don't expect most metric implementations to need it.
    ~Distance() = default;

  protected:
    Metric _distance_metric;
    size_t _alignment_factor = 8;
};


