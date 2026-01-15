"""
Extensive tests for IVF+PQ index implementation
"""
import numpy as np
import os
import shutil
import tempfile
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Import caliby
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import caliby


def compute_recall(ground_truth, results, k):
    """Compute recall@k"""
    if len(ground_truth) == 0:
        return 1.0
    hits = len(set(ground_truth[:k]) & set(results[:k]))
    return hits / min(k, len(ground_truth))


def brute_force_knn(data, query, k):
    """Compute exact k-NN using brute force"""
    distances = np.sum((data - query) ** 2, axis=1)
    indices = np.argsort(distances)[:k]
    return indices, distances[indices]


class TestIVFPQExtensive:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp(prefix='caliby_test_ivfpq_ext_')
        caliby.set_buffer_config(1.0, 4.0)  # 1GB physical, 4GB virtual
        caliby.open(self.test_dir, cleanup_if_exist=True)
        
        yield
        
        caliby.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # =========================================================================
    # Basic Functionality Tests
    # =========================================================================
    
    def test_different_dimensions(self):
        """Test with various dimensionalities"""
        for dim in [32, 64, 128, 256]:
            index = caliby.IVFPQIndex(
                max_elements=5000,
                dim=dim,
                num_clusters=32,
                num_subquantizers=dim // 4  # Ensure divisibility
            )
            
            np.random.seed(42)
            data = np.random.randn(1000, dim).astype(np.float32)
            
            index.train(data)
            index.add_points(data)
            
            assert index.get_count() == 1000
            assert index.get_dim() == dim
            
            # Test search
            labels, distances = index.search_knn(data[0], 10, 8)
            assert len(labels) == 10
    
    def test_different_cluster_counts(self):
        """Test with various numbers of clusters"""
        dim = 128
        for num_clusters in [16, 64, 128, 256]:
            index = caliby.IVFPQIndex(
                max_elements=5000,
                dim=dim,
                num_clusters=num_clusters,
                num_subquantizers=8
            )
            
            np.random.seed(42)
            data = np.random.randn(2000, dim).astype(np.float32)
            
            index.train(data)
            index.add_points(data)
            
            stats = index.get_stats()
            assert stats['num_clusters'] == num_clusters
            assert len(stats['list_sizes']) == num_clusters
    
    def test_different_subquantizer_counts(self):
        """Test with various numbers of subquantizers"""
        dim = 128
        for num_subq in [4, 8, 16, 32]:
            index = caliby.IVFPQIndex(
                max_elements=5000,
                dim=dim,
                num_clusters=64,
                num_subquantizers=num_subq
            )
            
            np.random.seed(42)
            data = np.random.randn(1000, dim).astype(np.float32)
            
            index.train(data)
            index.add_points(data)
            
            stats = index.get_stats()
            assert stats['num_subquantizers'] == num_subq

    # =========================================================================
    # Accuracy/Recall Tests
    # =========================================================================
    
    def test_recall_at_k(self):
        """Test recall accuracy compared to brute force"""
        dim = 128
        n_vectors = 10000
        n_queries = 100
        k = 10
        
        index = caliby.IVFPQIndex(
            max_elements=n_vectors + 1000,
            dim=dim,
            num_clusters=256,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        # Test with different nprobe values
        for nprobe in [1, 4, 16, 32, 64]:
            recalls = []
            for i in range(n_queries):
                # Brute force ground truth
                gt_indices, _ = brute_force_knn(data, queries[i], k)
                
                # IVF+PQ search
                labels, distances = index.search_knn(queries[i], k, nprobe)
                
                recall = compute_recall(gt_indices, labels, k)
                recalls.append(recall)
            
            avg_recall = np.mean(recalls)
            print(f"nprobe={nprobe}: avg recall@{k} = {avg_recall:.3f}")
            
            # IVF+PQ with M=8 on random data typically achieves 10-25% recall
            # This is lower than real data (like SIFT) due to lack of structure
            if nprobe >= 64:
                assert avg_recall > 0.08, f"Recall too low ({avg_recall}) for nprobe={nprobe}"
    
    def test_self_search_accuracy(self):
        """Test that searching for vectors in the index finds themselves"""
        dim = 128
        n_vectors = 5000
        
        index = caliby.IVFPQIndex(
            max_elements=n_vectors + 1000,
            dim=dim,
            num_clusters=128,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        # Search for first 100 vectors
        n_test = 100
        nprobe = 32
        hits = 0
        
        for i in range(n_test):
            labels, _ = index.search_knn(data[i], 10, nprobe)
            if i in labels[:5]:  # Should be in top 5
                hits += 1
        
        accuracy = hits / n_test
        print(f"Self-search accuracy (top-5): {accuracy:.2%}")
        assert accuracy > 0.8, f"Self-search accuracy too low: {accuracy}"

    # =========================================================================
    # Scale Tests
    # =========================================================================
    
    def test_large_scale_insert(self):
        """Test inserting a large number of vectors"""
        dim = 128
        n_vectors = 50000
        
        index = caliby.IVFPQIndex(
            max_elements=n_vectors + 10000,
            dim=dim,
            num_clusters=256,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        
        # Train with subset
        train_data = np.random.randn(10000, dim).astype(np.float32)
        index.train(train_data)
        
        # Insert in batches
        batch_size = 10000
        total_inserted = 0
        
        start_time = time.time()
        for batch_idx in range(n_vectors // batch_size):
            batch_data = np.random.randn(batch_size, dim).astype(np.float32)
            index.add_points(batch_data)
            total_inserted += batch_size
            
        elapsed = time.time() - start_time
        
        assert index.get_count() == n_vectors
        print(f"Inserted {n_vectors} vectors in {elapsed:.2f}s ({n_vectors/elapsed:.0f} vec/s)")
    
    def test_batch_search_consistency(self):
        """Test that batch search returns same results as single search"""
        dim = 128
        n_vectors = 5000
        n_queries = 50
        k = 10
        nprobe = 16
        
        index = caliby.IVFPQIndex(
            max_elements=n_vectors + 1000,
            dim=dim,
            num_clusters=128,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        # Single query results
        single_results = []
        for i in range(n_queries):
            labels, distances = index.search_knn(queries[i], k, nprobe)
            single_results.append((labels, distances))
        
        # Batch query results
        batch_labels, batch_distances = index.search_knn_parallel(queries, k, nprobe, num_threads=1)
        
        # Compare results (may have slight differences due to floating point)
        for i in range(n_queries):
            # At least top result should match
            assert single_results[i][0][0] == batch_labels[i][0], \
                f"Query {i}: single={single_results[i][0][0]}, batch={batch_labels[i][0]}"

    # =========================================================================
    # Edge Cases
    # =========================================================================
    
    def test_minimum_vectors(self):
        """Test with minimum number of vectors"""
        dim = 128
        num_clusters = 16
        
        index = caliby.IVFPQIndex(
            max_elements=1000,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        # Train with just enough vectors for clusters
        train_data = np.random.randn(num_clusters * 10, dim).astype(np.float32)
        index.train(train_data)
        
        # Add a few vectors
        data = np.random.randn(100, dim).astype(np.float32)
        index.add_points(data)
        
        labels, distances = index.search_knn(data[0], 5, 4)
        assert len(labels) <= 5
    
    def test_k_larger_than_index_size(self):
        """Test searching for more neighbors than vectors in index"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=1000,
            dim=dim,
            num_clusters=32,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(500, dim).astype(np.float32)
        
        index.train(data)
        
        # Add only 50 vectors
        index.add_points(data[:50])
        
        # Search for 100 (more than available)
        labels, distances = index.search_knn(data[0], 100, 32)
        
        # Should return at most 50 valid results
        valid_count = np.sum(labels >= 0)
        assert valid_count <= 50
    
    def test_high_nprobe(self):
        """Test with nprobe equal to num_clusters (exhaustive search)"""
        dim = 128
        num_clusters = 64
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(2000, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        # Search with nprobe = num_clusters (search all lists)
        labels, distances = index.search_knn(data[0], 10, num_clusters)
        
        assert len(labels) == 10
        assert 0 in labels  # Should definitely find the query itself
    
    def test_duplicate_vectors(self):
        """Test handling of duplicate vectors"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=64,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        base_data = np.random.randn(500, dim).astype(np.float32)
        
        # Create duplicates
        data = np.vstack([base_data, base_data, base_data])  # 1500 vectors, each duplicated 3x
        
        index.train(data)
        index.add_points(data)
        
        assert index.get_count() == 1500
        
        # Search should still work
        labels, distances = index.search_knn(base_data[0], 10, 16)
        assert len(labels) == 10

    # =========================================================================
    # Concurrent Operations Tests
    # =========================================================================
    
    def test_parallel_add(self):
        """Test parallel vector insertion"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=50000,
            dim=dim,
            num_clusters=128,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(20000, dim).astype(np.float32)
        
        index.train(data[:5000])
        
        # Add with multiple threads
        start_time = time.time()
        index.add_points(data, num_threads=4)
        elapsed = time.time() - start_time
        
        assert index.get_count() == 20000
        print(f"Parallel add: {20000/elapsed:.0f} vec/s with 4 threads")
    
    def test_parallel_search(self):
        """Test parallel search operations"""
        dim = 128
        n_vectors = 10000
        n_queries = 1000
        
        index = caliby.IVFPQIndex(
            max_elements=n_vectors + 1000,
            dim=dim,
            num_clusters=128,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        # Sequential search
        start_time = time.time()
        for i in range(n_queries):
            index.search_knn(queries[i], 10, 16)
        seq_time = time.time() - start_time
        
        # Parallel search
        start_time = time.time()
        index.search_knn_parallel(queries, 10, 16, num_threads=4)
        par_time = time.time() - start_time
        
        print(f"Sequential: {n_queries/seq_time:.0f} qps, Parallel: {n_queries/par_time:.0f} qps")
        # Parallel should be faster (or at least not much slower)
        assert par_time < seq_time * 2

    # =========================================================================
    # Statistics Tests
    # =========================================================================
    
    def test_list_balance(self):
        """Test that vectors are distributed across clusters"""
        dim = 128
        num_clusters = 64
        n_vectors = 6400  # 100 per cluster on average
        
        index = caliby.IVFPQIndex(
            max_elements=n_vectors + 1000,
            dim=dim,
            num_clusters=num_clusters,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(n_vectors, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        stats = index.get_stats()
        list_sizes = stats['list_sizes']
        
        # Check distribution
        min_size = min(list_sizes)
        max_size = max(list_sizes)
        avg_size = stats['avg_list_size']
        
        print(f"List sizes - min: {min_size}, max: {max_size}, avg: {avg_size:.1f}")
        
        # Lists shouldn't be too imbalanced (max shouldn't be more than 5x avg)
        assert max_size < avg_size * 5, f"Lists too imbalanced: max={max_size}, avg={avg_size}"
        
        # All clusters should have some vectors (with random data)
        empty_lists = sum(1 for s in list_sizes if s == 0)
        assert empty_lists < num_clusters * 0.1, f"Too many empty lists: {empty_lists}"
    
    def test_stats_accumulation(self):
        """Test that statistics accumulate correctly"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=64,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(2000, dim).astype(np.float32)
        
        index.train(data)
        index.add_points(data)
        
        # Reset stats
        index.reset_stats()
        stats = index.get_stats()
        assert stats['dist_comps'] == 0
        assert stats['lists_probed'] == 0
        assert stats['vectors_scanned'] == 0
        
        # Do some searches with stats=True
        n_searches = 10
        nprobe = 8
        for i in range(n_searches):
            index.search_knn(data[i], 10, nprobe, stats=True)
        
        stats = index.get_stats()
        assert stats['lists_probed'] == n_searches * nprobe
        assert stats['vectors_scanned'] > 0
        print(f"After {n_searches} searches: probed={stats['lists_probed']}, scanned={stats['vectors_scanned']}")

    # =========================================================================
    # Error Handling Tests
    # =========================================================================
    
    def test_add_before_train_fails(self):
        """Test that adding vectors before training raises error"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=64,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(100, dim).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="trained"):
            index.add_points(data)
    
    def test_search_before_train_fails(self):
        """Test that searching before training raises error"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=64,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        query = np.random.randn(dim).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="trained"):
            index.search_knn(query, 10, 8)
    
    def test_wrong_dimension_add_fails(self):
        """Test that adding vectors with wrong dimension raises error"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=64,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        train_data = np.random.randn(1000, dim).astype(np.float32)
        index.train(train_data)
        
        wrong_data = np.random.randn(100, dim + 10).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="dimension"):
            index.add_points(wrong_data)
    
    def test_wrong_dimension_search_fails(self):
        """Test that searching with wrong dimension raises error"""
        dim = 128
        
        index = caliby.IVFPQIndex(
            max_elements=5000,
            dim=dim,
            num_clusters=64,
            num_subquantizers=8
        )
        
        np.random.seed(42)
        data = np.random.randn(1000, dim).astype(np.float32)
        index.train(data)
        index.add_points(data)
        
        wrong_query = np.random.randn(dim + 10).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="dimension"):
            index.search_knn(wrong_query, 10, 8)


class TestIVFPQRecovery:
    """Test persistence and recovery"""
    
    def test_basic_recovery(self):
        """Test that index can be recovered after close/reopen.
        
        This test runs in a subprocess to ensure complete isolation
        from other tests' BufferManager state.
        """
        import subprocess
        import sys
        
        # Get the build directory path
        build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
        build_dir = os.path.abspath(build_dir)
        
        # Python script to run in subprocess for complete isolation
        test_script = f'''
import numpy as np
import sys
import os
sys.path.insert(0, "{build_dir}")
import caliby

dim = 128
n_vectors = 2000
unique_id = 60000

# Create and populate index with unique index_id
index = caliby.IVFPQIndex(
    max_elements=n_vectors + 1000,
    dim=dim,
    num_clusters=64,
    num_subquantizers=8,
    skip_recovery=True,  # Force fresh start
    index_id=unique_id,
)

np.random.seed(42)
data = np.random.randn(n_vectors, dim).astype(np.float32)

index.train(data)
index.add_points(data)

# Get search results before close
query = data[0]
labels_before, distances_before = index.search_knn(query, 10, 16)

# Flush all data to disk
index.flush()
caliby.flush_storage()

# Delete the index object to release resources
del index

# Create a new index with same parameters - should recover from disk
index2 = caliby.IVFPQIndex(
    max_elements=n_vectors + 1000,
    dim=dim,
    num_clusters=64,
    num_subquantizers=8,
    skip_recovery=False,  # Allow recovery
    index_id=unique_id,
)

# Verify recovery
assert index2.get_count() == n_vectors, f"Expected {{n_vectors}} vectors, got {{index2.get_count()}}"
assert index2.is_trained(), "Index should be trained after recovery"

# Search should give similar results
labels_after, distances_after = index2.search_knn(query, 10, 16)

# At least top result should match
assert labels_before[0] == labels_after[0], f"Top result changed: {{labels_before[0]}} vs {{labels_after[0]}}"

print("IVFPQ recovery test PASSED")
'''
        
        # Run the test in a subprocess for complete isolation
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        # Check if the subprocess succeeded
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            pytest.fail(f"IVFPQ recovery test failed in subprocess:\n{result.stderr}")
        
        assert "IVFPQ recovery test PASSED" in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])
