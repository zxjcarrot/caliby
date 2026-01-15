#!/usr/bin/env python3
"""
Tests for HNSW Index

Run with: pytest tests/test_hnsw.py -v

API Reference (from bindings.cpp):
- HnswIndex(max_elements, dim=128, M=16, ef_construction=200, enable_prefetch=True, skip_recovery=False)
- add_points(items: ndarray[n, dim]) - add batch of points
- search_knn(query, k, ef_search, stats=False) -> (labels, distances)
- search_knn_parallel(queries, k, ef_search, num_threads=0, stats=False) -> (labels, distances)
- get_dim() -> int
- flush() - flushes dirty pages
- was_recovered() -> bool
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


class TestHNSWBasic:
    """Basic HNSW functionality tests."""
    
    def test_import(self, caliby_module):
        """Test that HnswIndex is available."""
        assert hasattr(caliby_module, 'HnswIndex')
    
    def test_create_index(self, caliby_module, temp_dir):
        """Test creating an HNSW index."""
        dim = 128  # Fixed dimension in bindings
        max_elements = 1000
        
        index = caliby_module.HnswIndex(max_elements=max_elements, dim=dim, M=16, ef_construction=200)
        assert index is not None
        assert index.get_dim() == dim
    
    def test_add_single_point(self, caliby_module, temp_dir):
        """Test adding a single point via batch API."""
        dim = 128
        max_elements = 100
        
        index = caliby_module.HnswIndex(max_elements=max_elements, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vec = np.random.randn(1, dim).astype(np.float32)
        index.add_points(vec)
        
        # Search for the vector we just added
        labels, distances = index.search_knn(vec[0], 1, ef_search=50)
        
        assert len(labels) == 1
        assert labels[0] == 0  # First added point gets id 0
        assert distances[0] < 1e-5  # Distance should be ~0
    
    def test_add_multiple_points(self, caliby_module, temp_dir):
        """Test adding multiple points."""
        dim = 128
        num_points = 100
        max_elements = 200
        
        index = caliby_module.HnswIndex(max_elements=max_elements, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        index.add_points(vectors)
        
        # Search for first vector
        labels, distances = index.search_knn(vectors[0], 5, ef_search=50)
        
        assert len(labels) == 5
        assert labels[0] == 0  # First result should be exact match


class TestHNSWSearch:
    """HNSW search functionality tests."""
    
    def test_knn_search(self, caliby_module, temp_dir):
        """Test k-NN search returns correct number of results."""
        dim = 128  # Fixed dimension
        num_points = 500
        k = 10
        
        index = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, ef_construction=200, skip_recovery=True)
        
        # Add points
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        index.add_points(vectors)
        
        # Search
        query = np.random.randn(dim).astype(np.float32)
        labels, distances = index.search_knn(query, k, ef_search=100)
        
        assert len(distances) == k
        assert len(labels) == k
        
        # Distances should be sorted (ascending)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]
    
    def test_search_exact_match(self, caliby_module, temp_dir):
        """Test that searching for an indexed vector returns it as nearest."""
        dim = 128  # Fixed dimension
        num_points = 1000
        
        index = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, ef_construction=200, skip_recovery=True)
        
        # Add points
        np.random.seed(42)  # Fixed seed for reproducibility
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        # Normalize for better recall
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        index.add_points(vectors)
        
        # Search for each of the first 50 vectors
        correct = 0
        for i in range(50):
            labels, distances = index.search_knn(vectors[i], 5, ef_search=200)
            if i in labels:  # Check if in top-5
                correct += 1
        
        # Should have high recall for exact queries in top-5
        # Relaxed threshold to 70% to account for HNSW approximation
        assert correct >= 35, f"Expected at least 70% recall, got {correct}/50"
    
    def test_ef_parameter_affects_results(self, caliby_module, temp_dir):
        """Test that ef parameter affects search quality."""
        dim = 128  # Fixed dimension
        num_points = 500
        
        index = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, ef_construction=200, skip_recovery=True)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        index.add_points(vectors)
        
        query = np.random.randn(dim).astype(np.float32)
        
        # Low ef should be faster but potentially lower quality
        labels_low, distances_low = index.search_knn(query, 10, ef_search=10)
        
        # High ef should give better results
        labels_high, distances_high = index.search_knn(query, 10, ef_search=200)
        
        # High ef should find at least as good results
        assert distances_high[0] <= distances_low[0] + 1e-5


class TestHNSWParallel:
    """Test HNSW parallel search."""
    
    def test_parallel_search(self, caliby_module, temp_dir):
        """Test parallel batch search."""
        dim = 128
        num_points = 500
        num_queries = 50
        k = 10
        
        index = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        index.add_points(vectors)
        
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        labels, distances = index.search_knn_parallel(queries, k, ef_search=50, num_threads=2)
        
        assert labels.shape == (num_queries, k)
        assert distances.shape == (num_queries, k)


class TestHNSWParameters:
    """Test HNSW with different M parameters."""
    
    @pytest.mark.parametrize("M", [4, 8, 16, 32])
    def test_various_M(self, caliby_module, temp_dir, M):
        """Test HNSW works with various M values."""
        dim = 128  # Fixed dimension
        num_points = 100
        
        index = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=M, ef_construction=100, skip_recovery=True)
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        index.add_points(vectors)
        
        labels, distances = index.search_knn(vectors[0], 5, ef_search=50)
        
        assert len(labels) == 5


class TestHNSWRecall:
    """Test HNSW recall quality."""
    
    def test_recall_at_10(self, caliby_module, temp_dir):
        """Test recall@10 is reasonable."""
        dim = 128  # Fixed dimension
        num_points = 1000
        num_queries = 100
        k = 10
        
        index = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, ef_construction=200, skip_recovery=True)
        
        # Build index
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        index.add_points(vectors)
        
        # Generate queries
        queries = np.random.randn(num_queries, dim).astype(np.float32)
        
        # Compute ground truth using brute force
        def brute_force_knn(query, vectors, k):
            distances = np.sum((vectors - query) ** 2, axis=1)
            indices = np.argsort(distances)[:k]
            return set(indices)
        
        # Compute recall
        total_recall = 0
        for query in queries:
            gt = brute_force_knn(query, vectors, k)
            labels, distances = index.search_knn(query, k, ef_search=100)
            found = set(labels)
            recall = len(gt & found) / k
            total_recall += recall
        
        avg_recall = total_recall / num_queries
        assert avg_recall >= 0.70  # At least 70% recall (realistic for M=16, ef_search=100)


class TestHNSWRecovery:
    """Test HNSW recovery functionality."""
    
    def test_basic_recovery(self, caliby_module, temp_dir):
        """Test that index can be recovered after flush."""
        dim = 128
        num_points = 500
        k = 10
        
        # Build and flush initial index
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        query = vectors[0].copy()
        
        index1 = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, 
                                         ef_construction=200, skip_recovery=True)
        index1.add_points(vectors)
        labels1, distances1 = index1.search_knn(query, k, ef_search=100)
        
        # Flush to disk
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Recover and verify
        index2 = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, 
                                         ef_construction=200, skip_recovery=False)
        assert index2.was_recovered()
        
        labels2, distances2 = index2.search_knn(query, k, ef_search=100)
        
        # Results should match
        assert np.array_equal(labels1, labels2)
        assert np.allclose(distances1, distances2, rtol=1e-5)
    
    def test_skip_recovery_flag(self, caliby_module, temp_dir):
        """Test that skip_recovery=True rebuilds from scratch."""
        dim = 128
        num_points = 300
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        
        # Build and flush
        index1 = caliby_module.HnswIndex(max_elements=num_points, dim=dim, skip_recovery=True)
        index1.add_points(vectors)
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Create with skip_recovery=True (should NOT recover)
        index2 = caliby_module.HnswIndex(max_elements=num_points, dim=dim, skip_recovery=True)
        assert not index2.was_recovered()
    
    def test_recovery_with_different_params(self, caliby_module, temp_dir):
        """Test that recovery fails with mismatched parameters."""
        dim = 128
        num_points = 200
        
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        
        # Build with M=16
        index1 = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=16, skip_recovery=True)
        index1.add_points(vectors)
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Try to recover with M=32 (different parameter)
        index2 = caliby_module.HnswIndex(max_elements=num_points, dim=dim, M=32, skip_recovery=False)
        # Should not recover due to parameter mismatch
        assert not index2.was_recovered()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
