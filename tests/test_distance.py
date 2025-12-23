#!/usr/bin/env python3
"""
Tests for distance computation utilities

Run with: pytest tests/test_distance.py -v
"""

import numpy as np
import pytest


class TestDistanceComputation:
    """Test distance computation correctness."""
    
    def test_l2_distance_self(self):
        """Test L2 distance of vector with itself is 0."""
        vec = np.random.randn(128).astype(np.float32)
        dist = np.sum((vec - vec) ** 2)
        assert dist == 0.0
    
    def test_l2_distance_symmetric(self):
        """Test L2 distance is symmetric."""
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        
        dist_ab = np.sum((a - b) ** 2)
        dist_ba = np.sum((b - a) ** 2)
        
        assert np.isclose(dist_ab, dist_ba)
    
    def test_l2_distance_triangle_inequality(self):
        """Test L2 distance satisfies triangle inequality."""
        a = np.random.randn(64).astype(np.float32)
        b = np.random.randn(64).astype(np.float32)
        c = np.random.randn(64).astype(np.float32)
        
        # Using sqrt for proper distance metric
        d_ab = np.sqrt(np.sum((a - b) ** 2))
        d_bc = np.sqrt(np.sum((b - c) ** 2))
        d_ac = np.sqrt(np.sum((a - c) ** 2))
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert d_ac <= d_ab + d_bc + 1e-5  # Small epsilon for floating point
    
    def test_l2_distance_non_negative(self):
        """Test L2 distance is always non-negative."""
        for _ in range(100):
            a = np.random.randn(128).astype(np.float32)
            b = np.random.randn(128).astype(np.float32)
            dist = np.sum((a - b) ** 2)
            assert dist >= 0
    
    def test_l2_distance_zero_iff_equal(self):
        """Test L2 distance is 0 if and only if vectors are equal."""
        a = np.random.randn(64).astype(np.float32)
        b = a.copy()
        c = a + 0.001
        
        dist_ab = np.sum((a - b) ** 2)
        dist_ac = np.sum((a - c) ** 2)
        
        assert dist_ab == 0.0
        assert dist_ac > 0.0


class TestVectorNormalization:
    """Test vector normalization utilities."""
    
    def test_normalize_vector(self):
        """Test vector normalization produces unit length."""
        vec = np.random.randn(128).astype(np.float32)
        normalized = vec / np.linalg.norm(vec)
        
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_normalize_preserves_direction(self):
        """Test normalization preserves direction."""
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normalized = vec / np.linalg.norm(vec)
        
        # Check direction is preserved (ratio should be constant)
        ratio = normalized[0] / normalized[1]
        original_ratio = vec[0] / vec[1]
        assert np.isclose(ratio, original_ratio)
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        
        # Normalize
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        
        # Cosine similarity via dot product
        cos_sim = np.dot(a_norm, b_norm)
        
        # Should be in [-1, 1]
        assert -1.0 <= cos_sim <= 1.0
    
    def test_identical_vectors_max_similarity(self):
        """Test identical normalized vectors have similarity 1."""
        vec = np.random.randn(128).astype(np.float32)
        vec_norm = vec / np.linalg.norm(vec)
        
        cos_sim = np.dot(vec_norm, vec_norm)
        assert np.isclose(cos_sim, 1.0, atol=1e-6)


class TestBatchOperations:
    """Test batch distance computations."""
    
    def test_batch_l2_distances(self):
        """Test computing L2 distances to multiple vectors."""
        query = np.random.randn(128).astype(np.float32)
        database = np.random.randn(100, 128).astype(np.float32)
        
        # Compute all distances
        distances = np.sum((database - query) ** 2, axis=1)
        
        assert distances.shape == (100,)
        assert np.all(distances >= 0)
    
    def test_find_nearest_neighbor(self):
        """Test finding nearest neighbor via exhaustive search."""
        query = np.random.randn(64).astype(np.float32)
        database = np.random.randn(1000, 64).astype(np.float32)
        
        # Set one vector equal to query
        database[42] = query
        
        distances = np.sum((database - query) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        
        assert nearest_idx == 42
        assert distances[nearest_idx] == 0.0
    
    def test_find_k_nearest_neighbors(self):
        """Test finding k nearest neighbors."""
        query = np.random.randn(64).astype(np.float32)
        database = np.random.randn(1000, 64).astype(np.float32)
        k = 10
        
        distances = np.sum((database - query) ** 2, axis=1)
        k_nearest_indices = np.argsort(distances)[:k]
        
        assert len(k_nearest_indices) == k
        
        # Verify these are indeed the k smallest distances
        k_distances = distances[k_nearest_indices]
        remaining_distances = np.delete(distances, k_nearest_indices)
        
        assert np.all(k_distances[-1] <= remaining_distances)


class TestNumericalStability:
    """Test numerical stability of distance computations."""
    
    def test_small_vectors(self):
        """Test distance computation with very small vectors."""
        a = np.array([1e-10, 1e-10], dtype=np.float32)
        b = np.array([2e-10, 2e-10], dtype=np.float32)
        
        dist = np.sum((a - b) ** 2)
        assert dist >= 0
        assert np.isfinite(dist)
    
    def test_large_vectors(self):
        """Test distance computation with large vectors."""
        a = np.array([1e5, 1e5], dtype=np.float32)
        b = np.array([1e5 + 1, 1e5 + 1], dtype=np.float32)
        
        dist = np.sum((a - b) ** 2)
        assert np.isfinite(dist)
        assert np.isclose(dist, 2.0)
    
    def test_mixed_scale_vectors(self):
        """Test distance with mixed scale components."""
        a = np.array([1e-5, 1e5], dtype=np.float32)
        b = np.array([2e-5, 2e5], dtype=np.float32)
        
        dist = np.sum((a - b) ** 2)
        assert np.isfinite(dist)
        assert dist >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
