#!/usr/bin/env python3
"""
Recovery Tests for Caliby Indexes

Tests index persistence and recovery functionality for HNSW.
Run with: pytest tests/test_recovery.py -v
"""

import numpy as np
import pytest


class TestHNSWRecovery:
    """Test HNSW index recovery functionality."""
    
    def test_basic_recovery(self, caliby_module, temp_dir):
        """Test that index can be recovered after flush."""
        dim = 128
        num_points = 1000
        k = 10
        
        # Generate test data
        np.random.seed(42)
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query = vectors[0].copy()
        
        # Build and flush initial index
        index1 = caliby_module.HnswIndex(num_points, dim, M=16, 
                                         ef_construction=200, skip_recovery=True)
        index1.add_items(vectors)
        labels1, distances1 = index1.search_knn(query, k, ef_search=100)
        
        # Flush to disk
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Recover and verify
        index2 = caliby_module.HnswIndex(num_points, dim, M=16, 
                                         ef_construction=200, skip_recovery=False)
        assert index2.was_recovered(), "Index should have been recovered from disk"
        
        labels2, distances2 = index2.search_knn(query, k, ef_search=100)
        
        # Results should match exactly
        assert np.array_equal(labels1, labels2), "Labels should match after recovery"
        assert np.allclose(distances1, distances2, rtol=1e-5), "Distances should match after recovery"
    
    def test_skip_recovery_flag(self, caliby_module, temp_dir):
        """Test that skip_recovery=True rebuilds from scratch."""
        dim = 128
        num_points = 500
        
        np.random.seed(123)
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Build and flush
        index1 = caliby_module.HnswIndex(num_points, dim, skip_recovery=True)
        index1.add_items(vectors)
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Create with skip_recovery=True (should NOT recover)
        index2 = caliby_module.HnswIndex(num_points, dim, skip_recovery=True)
        assert not index2.was_recovered(), "Index should not recover with skip_recovery=True"
    
    def test_recovery_with_mismatched_params(self, caliby_module, temp_dir):
        """Test that recovery fails gracefully with mismatched parameters."""
        dim = 128
        num_points = 300
        
        np.random.seed(456)
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        
        # Build with M=16
        index1 = caliby_module.HnswIndex(num_points, dim, M=16, skip_recovery=True)
        index1.add_items(vectors)
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Try to recover with M=32 (different parameter)
        index2 = caliby_module.HnswIndex(num_points, dim, M=32, skip_recovery=False)
        # Should not recover due to parameter mismatch
        assert not index2.was_recovered(), "Index should not recover with mismatched M parameter"
    
    def test_multiple_recovery_cycles(self, caliby_module, temp_dir):
        """Test multiple recovery cycles maintain consistency."""
        dim = 128
        num_points = 500
        k = 5
        num_cycles = 3
        
        np.random.seed(789)
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query = vectors[0].copy()
        
        # Initial build
        index = caliby_module.HnswIndex(num_points, dim, skip_recovery=True)
        index.add_items(vectors)
        index.flush()
        caliby_module.flush_storage()
        
        initial_labels, initial_distances = index.search_knn(query, k, ef_search=100)
        del index
        
        # Test multiple recovery cycles
        for cycle in range(num_cycles):
            index = caliby_module.HnswIndex(num_points, dim, skip_recovery=False)
            
            assert index.was_recovered(), f"Recovery cycle {cycle+1} failed"
            
            labels, distances = index.search_knn(query, k, ef_search=100)
            
            assert np.array_equal(labels, initial_labels), \
                f"Labels differ after recovery cycle {cycle+1}"
            assert np.allclose(distances, initial_distances, rtol=1e-5), \
                f"Distances differ after recovery cycle {cycle+1}"
            
            # Flush for next cycle (except last)
            if cycle < num_cycles - 1:
                index.flush()
                caliby_module.flush_storage()
            del index
    
    def test_recovery_after_partial_build(self, caliby_module, temp_dir):
        """Test recovery when index was partially built."""
        dim = 128
        num_points_batch1 = 300
        num_points_batch2 = 200
        total_points = num_points_batch1 + num_points_batch2
        k = 5
        
        np.random.seed(111)
        vectors_batch1 = np.random.randn(num_points_batch1, dim).astype(np.float32)
        vectors_batch2 = np.random.randn(num_points_batch2, dim).astype(np.float32)
        
        # Build with first batch
        index1 = caliby_module.HnswIndex(total_points, dim, skip_recovery=True)
        index1.add_items(vectors_batch1)
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Recover and add second batch
        index2 = caliby_module.HnswIndex(total_points, dim, skip_recovery=False)
        assert index2.was_recovered(), "Should recover after partial build"
        
        index2.add_items(vectors_batch2)
        
        # Search should work with all data
        query = vectors_batch1[0]
        labels, distances = index2.search_knn(query, k, ef_search=100)
        
        assert len(labels) == k
        # The query is from batch1, so at least one result should be from batch1
        assert any(label < num_points_batch1 for label in labels), \
            "At least one result should be from the first batch"
    
    def test_recovery_preserves_graph_structure(self, caliby_module, temp_dir):
        """Test that recovery preserves the exact graph structure."""
        dim = 128
        num_points = 500
        
        np.random.seed(222)
        vectors = np.random.randn(num_points, dim).astype(np.float32)
        
        # Build index and get stats
        index1 = caliby_module.HnswIndex(num_points, dim, M=16, skip_recovery=True)
        index1.add_items(vectors)
        stats1 = index1.get_stats()
        index1.flush()
        caliby_module.flush_storage()
        del index1
        
        # Recover and check stats match
        index2 = caliby_module.HnswIndex(num_points, dim, M=16, skip_recovery=False)
        assert index2.was_recovered()
        stats2 = index2.get_stats()
        
        # Key graph structure metrics should match
        assert stats1['num_levels'] == stats2['num_levels'], "Number of levels should match"
        assert stats1['nodes_per_level'] == stats2['nodes_per_level'], "Nodes per level should match"
        assert stats1['links_per_level'] == stats2['links_per_level'], "Links per level should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
