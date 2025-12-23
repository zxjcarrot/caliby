#!/usr/bin/env python3
"""
Tests for DiskANN Index

Run with: pytest tests/test_diskann.py -v

API Reference (from bindings.cpp):
- DiskANN(dimensions, max_elements, R_max_degree=64, is_dynamic=False)
- BuildParams() - default constructor, attributes: L_build, alpha, num_threads
- SearchParams(L_search) - constructor takes L_search, attributes: L_search, beam_width

Methods:
- build(data, tags, params) - bulk load
- search(query, K, params) -> (distances, labels)
- search_knn_parallel(queries, K, params, num_threads=0)
- insert_point(point, tags, external_id) - for dynamic index
- lazy_delete(external_id)
- consolidate_deletes(params)
"""

import numpy as np
import pytest
import tempfile
import os
import sys

# Add build directory to path for local testing
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)


# Use shared fixtures from conftest.py instead of defining our own
# This ensures the heapfile is consistent across all test modules


class TestDiskANNBasic:
    """Basic DiskANN functionality tests."""
    
    def test_import(self, caliby_module):
        """Test that DiskANN classes are available."""
        assert hasattr(caliby_module, 'DiskANN')
        assert hasattr(caliby_module, 'BuildParams')
        assert hasattr(caliby_module, 'SearchParams')
    
    def test_create_build_params(self, caliby_module):
        """Test creating BuildParams."""
        params = caliby_module.BuildParams()
        # Should have default values
        assert params is not None
        # Check attributes exist
        assert hasattr(params, 'L_build')
        assert hasattr(params, 'alpha')
        assert hasattr(params, 'num_threads')
    
    def test_create_search_params(self, caliby_module):
        """Test creating SearchParams."""
        params = caliby_module.SearchParams(100)  # L_search = 100
        assert params is not None
        assert hasattr(params, 'L_search')
        assert hasattr(params, 'beam_width')
        assert params.L_search == 100
    
    def test_modify_build_params(self, caliby_module):
        """Test modifying BuildParams attributes."""
        params = caliby_module.BuildParams()
        params.L_build = 100
        params.alpha = 1.2
        params.num_threads = 4
        
        assert params.L_build == 100
        assert abs(params.alpha - 1.2) < 0.001
        assert params.num_threads == 4
    
    def test_modify_search_params(self, caliby_module):
        """Test modifying SearchParams attributes."""
        params = caliby_module.SearchParams(50)
        params.L_search = 100
        params.beam_width = 4
        
        assert params.L_search == 100
        assert params.beam_width == 4
    
    def test_create_index(self, caliby_module, temp_dir):
        """Test creating a DiskANN index."""
        dim = 128
        max_elements = 1000
        R_max_degree = 64
        is_dynamic = False
        
        # Create index - DiskANN(dimensions, max_elements, R_max_degree, is_dynamic)
        index = caliby_module.DiskANN(dim, max_elements, R_max_degree, is_dynamic)
        assert index is not None
        assert index.dimensions == dim
        assert index.R == R_max_degree


class TestDiskANNOperations:
    """DiskANN indexing and search tests."""
    
    def test_build_index(self, caliby_module, temp_dir):
        """Test building an index with data."""
        dim = 128
        num_points = 100
        max_elements = 200
        R_max_degree = 32
        
        index = caliby_module.DiskANN(dim, max_elements, R_max_degree, False)
        
        # Create data
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        # Tags: list of list of uint32
        tags = [[i] for i in range(num_points)]
        
        # Build params
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        build_params.alpha = 1.2
        build_params.num_threads = 1
        
        # Build the index
        index.build(vectors, tags, build_params)
    
    def test_search_after_build(self, caliby_module, temp_dir):
        """Test searching after building index."""
        dim = 128
        num_points = 200
        max_elements = 300
        R_max_degree = 32
        k = 10
        
        index = caliby_module.DiskANN(dim, max_elements, R_max_degree, False)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        tags = [[i] for i in range(num_points)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        build_params.alpha = 1.2
        
        index.build(vectors, tags, build_params)
        
        # Search
        search_params = caliby_module.SearchParams(100)
        query = np.random.randn(dim).astype(np.float32)
        
        labels, distances = index.search(query, k, search_params)
        
        assert len(distances) == k
        assert len(labels) == k


class TestDiskANNParallel:
    """Test DiskANN parallel search."""
    
    def test_parallel_search(self, caliby_module, temp_dir):
        """Test parallel batch search functionality."""
        dim = 128
        num_points = 300
        max_elements = 400
        R_max_degree = 32
        k = 10
        num_queries = 20
        
        index = caliby_module.DiskANN(dim, max_elements, R_max_degree, False)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        tags = [[i] for i in range(num_points)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        index.build(vectors, tags, build_params)
        
        # Parallel search
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        search_params = caliby_module.SearchParams(100)
        
        labels, distances = index.search_knn_parallel(queries, k, search_params, 1)
        
        assert distances.shape == (num_queries, k)
        assert labels.shape == (num_queries, k)


class TestDiskANNParameters:
    """Test DiskANN with different parameters."""
    
    @pytest.mark.parametrize("R", [16, 32, 64])
    def test_various_R_values(self, caliby_module, temp_dir, R):
        """Test DiskANN with various R (max degree) values."""
        dim = 128
        num_points = 100
        max_elements = 150
        
        index = caliby_module.DiskANN(dim, max_elements, R, False)
        assert index.R == R
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        tags = [[i] for i in range(num_points)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        index.build(vectors, tags, build_params)
        
        # Search
        search_params = caliby_module.SearchParams(100)
        query = np.random.randn(dim).astype(np.float32)
        labels, distances = index.search(query, 5, search_params)
        
        assert len(labels) == 5
    
    def test_L_search_affects_results(self, caliby_module, temp_dir):
        """Test that L_search parameter affects search quality."""
        dim = 128
        num_points = 500
        max_elements = 600
        R_max_degree = 64
        
        index = caliby_module.DiskANN(dim, max_elements, R_max_degree, False)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        tags = [[i] for i in range(num_points)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        index.build(vectors, tags, build_params)
        
        query = np.random.randn(dim).astype(np.float32)
        
        # Low L_search
        search_params_low = caliby_module.SearchParams(10)
        distances_low, labels_low = index.search(query, 10, search_params_low)
        
        # High L_search  
        search_params_high = caliby_module.SearchParams(200)
        distances_high, labels_high = index.search(query, 10, search_params_high)
        
        # Both should return results
        assert len(labels_low) >= 1
        assert len(labels_high) >= 1
        
        # Higher L_search should give better (smaller) or equal distances
        assert distances_high[0] <= distances_low[0] + 1e-5


class TestDiskANNRecall:
    """Test DiskANN recall quality."""
    
    def test_recall_at_10(self, caliby_module, temp_dir):
        """Test recall@10 is reasonable."""
        dim = 128
        num_points = 500
        max_elements = 600
        num_queries = 50
        k = 10
        
        index = caliby_module.DiskANN(dim, max_elements, 64, False)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        tags = [[i] for i in range(num_points)]
        
        build_params = caliby_module.BuildParams()
        build_params.L_build = 100
        build_params.alpha = 1.2
        index.build(vectors, tags, build_params)
        
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        
        # Compute ground truth
        def brute_force_knn(query, vectors, k):
            distances = np.sum((vectors - query) ** 2, axis=1)
            indices = np.argsort(distances)[:k]
            return set(indices)
        
        search_params = caliby_module.SearchParams(100)
        
        total_recall = 0
        for query in queries:
            gt = brute_force_knn(query, vectors, k)
            labels, distances = index.search(query, k, search_params)
            found = set(labels)
            recall = len(gt & found) / k
            total_recall += recall
        
        avg_recall = total_recall / num_queries
        # DiskANN should achieve decent recall
        assert avg_recall >= 0.5, f"DiskANN recall too low: {avg_recall}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
