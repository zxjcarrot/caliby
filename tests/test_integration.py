#!/usr/bin/env python3
"""
Integration tests for Caliby

These tests verify the complete workflow of building and querying indexes.

Run with: pytest tests/test_integration.py -v

API Reference:
- HnswIndex(max_elements, dim=128, M=16, ef_construction=200, enable_prefetch=True, skip_recovery=False)
  - add_points(items: ndarray[n, dim])
  - search_knn(query, k, ef_search, stats=False) -> (labels, distances)
- DiskANN(dimensions, max_elements, R_max_degree, is_dynamic)
  - build(data, tags, params)
  - search(query, k, params) -> (distances, labels)
- BuildParams() - L_build, alpha, num_threads
- SearchParams(L_search) - L_search, beam_width
"""

import numpy as np
import pytest
import tempfile
import os
import sys
import time

# Add build directory to path for local testing
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)


# Use shared fixtures from conftest.py instead of defining our own
# This ensures the heapfile is consistent across all test modules


class TestEndToEndHNSW:
    """End-to-end tests for HNSW index."""
    
    def test_complete_workflow(self, caliby_module, temp_dir):
        """Test complete workflow: create, add, search."""
        dim = 128  # Fixed dimension
        num_vectors = 1000
        k = 10
        
        # 1. Create index
        index = caliby_module.HnswIndex(max_elements=num_vectors, dim=dim, M=16, ef_construction=200, skip_recovery=True)
        
        # 2. Generate and add vectors
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        index.add_points(vectors)
        
        # 3. Search
        # Query with a known vector
        query = vectors[0].copy()
        labels, distances = index.search_knn(query, k, ef_search=100)
        
        # 4. Verify results
        assert len(labels) == k
        assert labels[0] == 0  # First result should be exact match
        assert distances[0] < 1e-5  # Distance should be ~0
        
        # Query with random vector
        random_query = np.random.randn(dim).astype(np.float32)
        random_query = random_query / np.linalg.norm(random_query)
        
        labels, distances = index.search_knn(random_query, k, ef_search=100)
        assert len(labels) == k
        
        # Distances should be sorted
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]
    
    def test_batch_queries(self, caliby_module, temp_dir):
        """Test running many queries in sequence."""
        dim = 128  # Fixed dimension
        num_vectors = 500
        num_queries = 100
        k = 10
        
        # Build index
        index = caliby_module.HnswIndex(max_elements=num_vectors, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        
        # Run batch queries
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        
        all_results = []
        for query in queries:
            labels, distances = index.search_knn(query, k, ef_search=50)
            all_results.append((labels, distances))
        
        assert len(all_results) == num_queries
        
        # Verify all queries returned k results
        for labels, distances in all_results:
            assert len(labels) == k


class TestEndToEndDiskANN:
    """End-to-end tests for DiskANN index."""
    
    def test_complete_workflow(self, caliby_module, temp_dir):
        """Test complete workflow: create, build, search."""
        dim = 128
        num_vectors = 500
        max_elements = 600
        k = 10
        
        # 1. Create index - DiskANN(dimensions, max_elements, R_max_degree, is_dynamic)
        index = caliby_module.DiskANN(dim, max_elements, 64, False)
        
        # 2. Generate vectors
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        tags = [[i] for i in range(num_vectors)]
        
        # 3. Build index
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        build_params.alpha = 1.2
        index.build(vectors, tags, build_params)
        
        # 4. Search
        search_params = caliby_module.SearchParams(100)
        
        # Query with a random vector
        random_query = np.random.randn(dim).astype(np.float32)
        random_query = random_query / np.linalg.norm(random_query)
        
        labels, distances = index.search(random_query, k, search_params)
        
        # 5. Verify results
        assert len(labels) == k
        # Distances should be sorted (ascending)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]
        
        # Additional query
        another_query = np.random.randn(dim).astype(np.float32)
        another_query = another_query / np.linalg.norm(another_query)
        
        labels2, distances2 = index.search(another_query, k, search_params)
        assert len(labels2) == k
    
    def test_batch_queries(self, caliby_module, temp_dir):
        """Test batch search with DiskANN."""
        dim = 128
        num_vectors = 300
        max_elements = 400
        num_queries = 50
        k = 10
        
        # Build index
        index = caliby_module.DiskANN(dim, max_elements, 32, False)
        
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        tags = [[i] for i in range(num_vectors)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        index.build(vectors, tags, build_params)
        
        # Batch search
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        search_params = caliby_module.SearchParams(100)
        
        labels, distances = index.search_knn_parallel(queries, k, search_params, 1)
        
        assert distances.shape == (num_queries, k)
        assert labels.shape == (num_queries, k)


class TestPerformance:
    """Basic performance tests."""
    
    def test_hnsw_throughput(self, caliby_module, temp_dir):
        """Measure HNSW query throughput."""
        dim = 128
        num_vectors = 1000
        num_queries = 1000
        k = 10
        
        # Build index
        index = caliby_module.HnswIndex(max_elements=num_vectors, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        
        # Benchmark queries
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        
        start = time.time()
        for query in queries:
            index.search_knn(query, k, ef_search=50)
        elapsed = time.time() - start
        
        qps = num_queries / elapsed
        print(f"\nHNSW QPS: {qps:.0f} queries/second")
        
        # Should be reasonably fast
        assert qps > 100, f"HNSW too slow: {qps} QPS"
    
    def test_diskann_throughput(self, caliby_module, temp_dir):
        """Measure DiskANN query throughput."""
        dim = 128
        num_vectors = 1000
        max_elements = 1100
        num_queries = 500
        k = 10
        
        # Build index
        index = caliby_module.DiskANN(dim, max_elements, 32, False)
        
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        tags = [[i] for i in range(num_vectors)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        index.build(vectors, tags, build_params)
        
        # Benchmark queries
        search_params = caliby_module.SearchParams(50)
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        
        start = time.time()
        for query in queries:
            index.search(query, k, search_params)
        elapsed = time.time() - start
        
        qps = num_queries / elapsed
        print(f"\nDiskANN QPS: {qps:.0f} queries/second")
        
        # Should be reasonably fast
        assert qps > 50, f"DiskANN too slow: {qps} QPS"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_vector(self, caliby_module, temp_dir):
        """Test with just one vector."""
        dim = 128
        
        index = caliby_module.HnswIndex(max_elements=10, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vec = np.random.randn(1, dim).astype(np.float32)
        index.add_points(vec)
        
        labels, distances = index.search_knn(vec[0], 1, ef_search=50)
        
        assert len(labels) == 1
        assert labels[0] == 0
    
    def test_search_k_larger_than_index(self, caliby_module, temp_dir):
        """Test searching for more neighbors than exist."""
        dim = 128
        num_vectors = 5
        
        index = caliby_module.HnswIndex(max_elements=10, dim=dim, M=4, ef_construction=50, skip_recovery=True)
        
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        
        query = np.random.randn(dim).astype(np.float32)
        
        # Request more neighbors than exist
        labels, distances = index.search_knn(query, 20, ef_search=50)
        
        # Should return at most num_vectors results
        assert len(labels) <= num_vectors
    
    def test_zero_vector(self, caliby_module, temp_dir):
        """Test with zero vectors."""
        dim = 128
        
        index = caliby_module.HnswIndex(max_elements=10, dim=dim, M=8, ef_construction=50, skip_recovery=True)
        
        # Add zero vector and normal vector
        vecs = np.zeros((2, dim), dtype=np.float32)
        vecs[1] = np.random.randn(dim)
        index.add_points(vecs)
        
        labels, distances = index.search_knn(vecs[0], 1, ef_search=50)
        
        assert len(labels) == 1
        assert labels[0] == 0


class TestFlushStorage:
    """Test storage flushing."""
    
    def test_flush_storage(self, caliby_module, temp_dir):
        """Test that flush_storage can be called."""
        dim = 128
        num_vectors = 100
        
        index = caliby_module.HnswIndex(max_elements=num_vectors, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        
        # This should not raise an error
        caliby_module.flush_storage()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
