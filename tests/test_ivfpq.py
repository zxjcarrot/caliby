"""
Test IVF+PQ index implementation
"""
import numpy as np
import os
import shutil
import tempfile
import pytest

# Import caliby
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import caliby


class TestIVFPQ:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix='caliby_test_ivfpq_')
        # Set a small buffer pool for testing
        caliby.set_buffer_config(0.5, 2.0)  # 0.5GB physical, 2GB virtual
        caliby.open(self.test_dir, cleanup_if_exist=True)
        
        yield
        
        # Cleanup
        caliby.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_create_index(self):
        """Test creating an IVF+PQ index"""
        dim = 128
        max_elements = 10000
        num_clusters = 64
        num_subquantizers = 8
        
        index = caliby.IVFPQIndex(
            max_elements=max_elements,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=num_subquantizers
        )
        
        assert index.get_dim() == dim
        assert index.get_count() == 0
        assert not index.is_trained()
    
    def test_train_and_add(self):
        """Test training and adding vectors"""
        dim = 128
        max_elements = 10000
        num_clusters = 64
        num_subquantizers = 8
        
        index = caliby.IVFPQIndex(
            max_elements=max_elements,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=num_subquantizers
        )
        
        # Generate training data
        np.random.seed(42)
        n_train = 5000
        training_data = np.random.randn(n_train, dim).astype(np.float32)
        
        # Train the index
        index.train(training_data)
        assert index.is_trained()
        
        # Add vectors
        n_vectors = 1000
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        
        assert index.get_count() == n_vectors
    
    def test_search(self):
        """Test searching the index"""
        dim = 128
        max_elements = 10000
        num_clusters = 64
        num_subquantizers = 8
        
        index = caliby.IVFPQIndex(
            max_elements=max_elements,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=num_subquantizers
        )
        
        # Generate and add data
        np.random.seed(42)
        n_vectors = 2000
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        
        # Train with all vectors
        index.train(vectors)
        index.add_points(vectors)
        
        # Search for nearest neighbors of first vector
        k = 10
        nprobe = 8
        query = vectors[0]
        labels, distances = index.search_knn(query, k, nprobe)
        
        # The first result should be the query itself (distance ~0)
        assert len(labels) == k
        assert len(distances) == k
        # Due to PQ compression, exact match may not be perfect
        # But we expect the query itself to be among top results
        assert 0 in labels[:3]  # Query should be in top 3
    
    def test_batch_search(self):
        """Test batch search"""
        dim = 128
        max_elements = 10000
        num_clusters = 64
        num_subquantizers = 8
        
        index = caliby.IVFPQIndex(
            max_elements=max_elements,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=num_subquantizers
        )
        
        # Generate and add data
        np.random.seed(42)
        n_vectors = 2000
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        
        index.train(vectors)
        index.add_points(vectors)
        
        # Batch search
        n_queries = 10
        k = 5
        nprobe = 8
        queries = vectors[:n_queries]
        labels, distances = index.search_knn_parallel(queries, k, nprobe, num_threads=4)
        
        assert labels.shape == (n_queries, k)
        assert distances.shape == (n_queries, k)
        
        # Each query should find itself
        for i in range(n_queries):
            assert i in labels[i][:3]  # Query i should be in top 3
    
    def test_stats(self):
        """Test getting statistics"""
        dim = 128
        max_elements = 10000
        num_clusters = 64
        num_subquantizers = 8
        
        index = caliby.IVFPQIndex(
            max_elements=max_elements,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=num_subquantizers
        )
        
        np.random.seed(42)
        n_vectors = 1000
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        
        index.train(vectors)
        index.add_points(vectors)
        
        # Get stats
        stats = index.get_stats()
        
        assert stats['num_clusters'] == num_clusters
        assert stats['num_subquantizers'] == num_subquantizers
        assert len(stats['list_sizes']) == num_clusters
        
        # Reset and verify
        index.reset_stats()
        stats = index.get_stats()
        assert stats['dist_comps'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
