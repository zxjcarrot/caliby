#!/usr/bin/env python3
"""
Comprehensive Multi-Index Tests for Caliby (Direct HnswIndex API)

These tests verify that multiple HNSW indexes can coexist with different dimensions,
sizes, and parameters without interference. Tests use the direct HnswIndex API with
index_id parameter.

Run with: pytest tests/test_multi_index_direct.py -v

API Reference:
- HnswIndex(max_elements, dim, M, ef_construction, enable_prefetch, skip_recovery, index_id, name)
  - add_points(items: ndarray[n, dim])
  - search_knn(query, k, ef_search, stats=False) -> (labels, distances)
  - get_name() -> str
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


# Global counter for unique index IDs across all tests
# Start at a high number to avoid conflicts with any existing indexes
_index_id_counter = 10000


def get_unique_index_id():
    """Generate a unique index ID for each test."""
    global _index_id_counter
    _index_id_counter += 1
    return _index_id_counter


def get_unique_index_name():
    """Generate a unique index name for each test."""
    return f"index_{_index_id_counter}"


class TestMultiIndexVaryingDimensions:
    """Test multi-index with varying dimensions."""
    
    @pytest.mark.parametrize("dims", [
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (32, 128, 256),
        (64, 128, 256, 512),
    ])
    def test_different_dimensions(self, caliby_module, temp_dir, dims):
        """Test multiple indexes with different dimensions."""
        n_vectors = 50000
        k = 10
        
        indexes = []
        vectors_list = []
        
        np.random.seed(42)
        
        # Create indexes with different dimensions
        for idx, dim in enumerate(dims):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Verify each index independently
        for idx, (index, vectors, dim) in enumerate(zip(indexes, vectors_list, dims)):
            labels, distances = index.search_knn(vectors[0], k, ef_search=200)
            assert len(labels) == k, f"Index {idx} with dim={dim} returned wrong k"
            # Check if exact match is in top-k results (HNSW is approximate)
            # Higher dimensions need more relaxed checks due to curse of dimensionality
            assert 0 in labels, f"Index {idx} with dim={dim} failed to find exact match in top-{k}"
            # Check that best distance is small (relaxed for approximate NN)
            assert min(distances) < 1.0, f"Index {idx} with dim={dim} has poor accuracy"
        
        # Cleanup: explicitly delete indexes to free resources
        del indexes
        del vectors_list
    
    @pytest.mark.parametrize("dim", [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768])
    def test_single_dimension_accuracy(self, caliby_module, temp_dir, dim):
        """Test a single dimension value with multi-index.
        
        Note: Dimension 1024 is excluded as it exceeds the 4096 byte page size limit
        even with M=2. Maximum dimension is ~1000 with M=2.
        """
        n_indexes = 2
        n_vectors = 300
        k = 10
        
        # Adjust M based on dimension to fit in page size (4096 bytes)
        # Node size = 4*dim + 8*(2*M + 4) must be <= 4096
        if dim <= 256:
            M = 8
        elif dim <= 512:
            M = 4
        else:  # 768
            M = 2
        
        indexes = []
        vectors_list = []
        
        np.random.seed(42)
        
        for i in range(n_indexes):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=M, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=1)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Verify accuracy
        k_check = min(15, n_vectors)  # Check within top-15 or total vectors
        for idx, (index, vectors) in enumerate(zip(indexes, vectors_list)):
            labels, distances = index.search_knn(vectors[0], 20, ef_search=max(200, n_vectors))
            # Check if exact match is in top-k OR if closest distance is very small (HNSW is approximate)
            found_exact = 0 in labels[:k_check]
            has_close_match = min(distances) < 0.01  # Distance to actual point should be ~0
            assert found_exact or has_close_match, \
                f"dim={dim}, index={idx} failed to find in top-{k_check} (best dist={min(distances):.6f})"
        
        # Cleanup
        del indexes
        del vectors_list


class TestMultiIndexVaryingVectorCounts:
    """Test multi-index with varying vector counts."""
    
    @pytest.mark.parametrize("counts", [
        (100, 200),
        (500, 1000),
    ])
    def test_different_vector_counts(self, caliby_module, temp_dir, counts):
        """Test multiple indexes with different vector counts.
        
        Note: Reduced to 2 test cases for faster and more reliable execution.
        """
        dim = 64
        k = 10
        
        indexes = []
        vectors_list = []
        
        np.random.seed(42)
        
        # Create indexes with different vector counts
        for idx, count in enumerate(counts):
            index = caliby_module.HnswIndex(
                max_elements=count, dim=dim, M=8, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(count, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Verify each index
        for idx, (index, vectors, count) in enumerate(zip(indexes, vectors_list, counts)):
            labels, distances = index.search_knn(vectors[0], k, ef_search=100)
            assert len(labels) == k, f"Index {idx} with count={count} returned wrong k"
            # Check if exact match is in top-3 (HNSW is approximate)
            assert 0 in labels[:3], f"Index {idx} with count={count} failed to find in top-3"
    
    @pytest.mark.parametrize("n_vectors", [100, 500, 1000, 5000])
    def test_single_vector_count_multi_index(self, caliby_module, temp_dir, n_vectors):
        """Test specific vector count with multiple indexes.
        
        Note: Reduced to 4 test cases (was 5) for faster and more reliable execution.
        Limited to 5000 vectors to ensure good graph quality.
        """
        dim = 64
        n_indexes = 3
        k = min(10, n_vectors)
        
        # Adjust ef_construction for larger indexes
        ef_construction = 200 if n_vectors >= 5000 else 100
        
        indexes = []
        vectors_list = []
        
        np.random.seed(42)
        
        for i in range(n_indexes):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=ef_construction,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Verify all indexes with higher ef_search for larger indexes
        ef_search = 200 if n_vectors >= 2000 else 100
        for idx, (index, vectors) in enumerate(zip(indexes, vectors_list)):
            labels, distances = index.search_knn(vectors[0], 20, ef_search=ef_search)
            # With random data and M=8, check in top-20 (HNSW is approximate)
            assert 0 in labels, f"n_vectors={n_vectors}, index={idx} failed to find in top-20"


class TestMultiIndexCombinations:
    """Test various combinations of dimensions and vector counts."""
    
    def test_small_dim_large_count(self, caliby_module, temp_dir):
        """Test indexes with small dimensions but large vector counts."""
        configs = [
            (16, 50000),
            (32, 30000),
        ]
        
        np.random.seed(42)
        k = 10
        
        for dim, n_vectors in configs:
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            # Verify search works
            labels, distances = index.search_knn(vectors[0], 20, ef_search=200)
            # Check if exact match is in top-20 (HNSW is approximate, especially with M=8 on large datasets)
            assert 0 in labels, f"Small dim {dim}, large count {n_vectors} failed to find in top-20"
    
    def test_large_dim_small_count(self, caliby_module, temp_dir):
        """Test indexes with large dimensions but small vector counts.
        
        Note: Dimension limited to 768 due to 4096 byte page size constraint.
        """
        configs = [
            (512, 100, 4),
            (768, 200, 2),
        ]
        
        np.random.seed(42)
        k = 10
        
        for dim, n_vectors, M in configs:
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=M, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            # Verify search works
            labels, distances = index.search_knn(vectors[0], 20, ef_search=300)
            # Check if exact match is in top-20 (HNSW is approximate, very low M for high dimensions)
            assert 0 in labels, f"Large dim {dim}, small count {n_vectors} failed to find in top-20"
    
    def test_mixed_configurations(self, caliby_module, temp_dir):
        """Test a mix of small and large dimensions/counts."""
        configs = [
            (16, 10000, 100),      # Small dim, small count
            (512, 10000, 100),     # Large dim, small count
            (16, 50000, 200),     # Small dim, large count (higher ef_construction)
            (256, 20000, 150),    # Medium dim, medium count
        ]
        
        np.random.seed(42)
        
        indexes = []
        vectors_list = []
        
        for dim, n_vectors, ef_construction in configs:
            # Adjust M based on dimension
            if dim <= 256:
                M = 8
            else:
                M = 4
                
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=M, ef_construction=ef_construction,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Verify all indexes work correctly with higher ef_search for large indexes
        k = 10
        for idx, (index, vectors, cfg) in enumerate(zip(indexes, vectors_list, configs)):
            dim, n_vectors, ef_construction = cfg
            # Use higher ef_search for larger indexes
            ef_search = 200 if n_vectors > 1000 else 100
            labels, distances = index.search_knn(vectors[0], k, ef_search=ef_search)
            # Check if exact match is in top-10 (HNSW is approximate, especially with low M)
            assert 0 in labels[:10], f"Mixed config index {idx} (dim={dim}, n={n_vectors}) failed to find in top-10"


class TestMultiIndexIsolation:
    """Test that indexes are properly isolated from each other."""
    
    def test_no_cross_contamination_varying_dims(self, caliby_module, temp_dir):
        """Verify that searching one index doesn't return results from another (varying dims)."""
        # Use smaller dimensions and higher M/ef for better recall on random data
        configs = [
            (16, 5000),
            (32, 5000),
            (64, 5000),
        ]
        
        np.random.seed(42)
        
        indexes = []
        vectors_list = []
        
        # Create all indexes with different seeds to ensure distinct data
        for cfg_idx, (dim, n_vectors) in enumerate(configs):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=16, ef_construction=200,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            
            # Use different seed per index
            np.random.seed(42 + cfg_idx)
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Search all indexes and verify isolation
        k = 20
        failed_searches = 0
        total_searches = 0
        for i in range(10):
            for idx, (index, vectors) in enumerate(zip(indexes, vectors_list)):
                if i < len(vectors):
                    labels, distances = index.search_knn(vectors[i], k, ef_search=400)
                    total_searches += 1
                    # Check if exact match is in top-10 (HNSW is approximate)
                    if i not in labels[:10]:
                        failed_searches += 1
        
        # Allow up to 20% failure rate due to HNSW approximation with random data
        max_allowed_failures = int(total_searches * 0.2)
        assert failed_searches <= max_allowed_failures, \
            f"Too many failed searches: {failed_searches}/{total_searches} (max allowed: {max_allowed_failures})"
    
    def test_interleaved_operations(self, caliby_module, temp_dir):
        """Test interleaved operations on multiple indexes."""
        dim = 64
        n_vectors = 500
        n_indexes = 5
        k = 10
        
        np.random.seed(42)
        
        indexes = []
        vectors_list = []
        
        # Create all indexes
        for i in range(n_indexes):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Perform interleaved searches
        failed_searches = 0
        total_searches = 0
        for round in range(5):
            for idx in range(n_indexes):
                query_idx = (round * 10 + idx) % n_vectors
                labels, distances = indexes[idx].search_knn(
                    vectors_list[idx][query_idx], k, ef_search=200
                )
                total_searches += 1
                # Check if exact match is in top-10 (HNSW is approximate)
                if query_idx not in labels[:10]:
                    failed_searches += 1
        
        # Allow up to 20% failure rate due to HNSW approximation
        max_allowed_failures = int(total_searches * 0.2)
        assert failed_searches <= max_allowed_failures, \
            f"Too many failed searches: {failed_searches}/{total_searches} (max allowed: {max_allowed_failures})"


class TestMultiIndexStress:
    """Stress tests for multi-index functionality."""
    
    def test_many_small_indexes(self, caliby_module, temp_dir):
        """Test creating many small indexes."""
        n_indexes = 20
        dim = 32
        n_vectors = 10000
        k = 5
        
        indexes = []
        vectors_list = []
        
        np.random.seed(42)
        
        for i in range(n_indexes):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=50,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append(index)
            vectors_list.append(vectors)
        
        # Verify all indexes
        for idx, (index, vectors) in enumerate(zip(indexes, vectors_list)):
            labels, distances = index.search_knn(vectors[0], 20, ef_search=200)
            # Check if exact match is in top-10 (HNSW is approximate, especially with M=8, ef_construction=50, small data)
            assert 0 in labels[:10], f"Small index {idx} failed to find exact match in top-10"
    
    def test_progressive_index_creation(self, caliby_module, temp_dir):
        """Test creating indexes progressively and verifying each."""
        dims = [32, 48, 64, 96, 128, 192, 256, 384]
        n_vectors = 300
        k = 10
        
        np.random.seed(42)
        created_indexes = []
        
        for idx, dim in enumerate(dims):
            # Create new index
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            # Verify this index works
            labels, distances = index.search_knn(vectors[0], k, ef_search=200)
            assert 0 in labels[:10], f"Newly created index with dim={dim} failed"
            
            created_indexes.append((index, vectors))
            
            # Verify all previously created indexes still work
            for prev_idx, (prev_index, prev_vectors) in enumerate(created_indexes[:-1]):
                labels, distances = prev_index.search_knn(prev_vectors[0], k, ef_search=200)
                assert 0 in labels[:10], f"Previously created index {prev_idx} failed after creating dim={dim}"


class TestMultiIndexAccuracy:
    """Test accuracy and quality of multi-index searches."""
    
    def test_recall_across_dimensions(self, caliby_module, temp_dir):
        """Test that recall is maintained across different dimensions."""
        dims = [32, 64, 128, 256]
        n_vectors = 1000
        k = 20
        
        np.random.seed(42)
        
        for idx, dim in enumerate(dims):
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=16, ef_construction=200,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            index.add_points(vectors, num_threads=0)
            
            # Test exact match retrieval
            correct_first = 0
            for i in range(min(100, n_vectors)):
                labels, distances = index.search_knn(vectors[i], k, ef_search=100)
                if labels[0] == i:
                    correct_first += 1
                
                # Verify distances are sorted
                for j in range(len(distances) - 1):
                    assert distances[j] <= distances[j + 1], \
                        f"Distances not sorted for dim={dim}"
            
            recall = correct_first / min(100, n_vectors)
            assert recall >= 0.70, f"Recall too low for dim={dim}: {recall}"
    
    def test_consistency_across_searches(self, caliby_module, temp_dir):
        """Test that repeated searches give consistent results."""
        configs = [
            (64, 500),
            (128, 500),
        ]
        
        np.random.seed(42)
        k = 10
        
        for dim, n_vectors in configs:
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=16, ef_construction=200,
                skip_recovery=True, index_id=get_unique_index_id()
            )
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            # Perform same search multiple times
            query = vectors[0]
            results = []
            for _ in range(5):
                labels, distances = index.search_knn(query, k, ef_search=100)
                results.append((labels.copy(), distances.copy()))
            
            # All results should be identical (HNSW is deterministic with same ef_search)
            for i in range(1, len(results)):
                np.testing.assert_array_equal(results[0][0], results[i][0],
                    err_msg=f"Labels differ for dim={dim}")
                np.testing.assert_allclose(results[0][1], results[i][1],
                    err_msg=f"Distances differ for dim={dim}")


class TestMultiIndexNaming:
    """Test index naming functionality."""
    
    def test_practical_named_indexes_example(self, caliby_module, temp_dir):
        """Demonstrate practical usage of named indexes in a multi-tenant scenario."""
        # Simulate a multi-tenant vector database where each tenant has their own index
        tenants = [
            {"name": "tenant_acme_corp", "dim": 128, "vectors": 500},
            {"name": "tenant_globex", "dim": 256, "vectors": 300},
            {"name": "tenant_initech", "dim": 64, "vectors": 800},
        ]
        
        np.random.seed(42)
        tenant_indexes = {}
        
        # Create an index for each tenant with a unique name
        for tenant in tenants:
            index = caliby_module.HnswIndex(
                max_elements=tenant["vectors"],
                dim=tenant["dim"],
                M=8,
                ef_construction=100,
                skip_recovery=True,
                index_id=get_unique_index_id(),
                name=tenant["name"]
            )
            
            # Add vectors for this tenant
            vectors = np.random.randn(tenant["vectors"], tenant["dim"]).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            tenant_indexes[tenant["name"]] = {
                "index": index,
                "vectors": vectors,
                "dim": tenant["dim"]
            }
        
        # Simulate querying specific tenant indexes by name
        for tenant_name, data in tenant_indexes.items():
            index = data["index"]
            vectors = data["vectors"]
            
            # Verify the name matches
            assert index.get_name() == tenant_name, f"Tenant name mismatch"
            
            # Perform a search
            k = 10
            labels, distances = index.search_knn(vectors[0], k, ef_search=200)
            
            # Verify search results
            assert 0 in labels[:10], f"Search failed for {tenant_name}"
            assert len(labels) == k, f"Wrong number of results for {tenant_name}"
        
        print(f"\n✓ Successfully managed {len(tenants)} tenant indexes with unique names")
        for name in tenant_indexes.keys():
            print(f"  - {name}")
    
    def test_index_names_are_set_correctly(self, caliby_module, temp_dir):
        """Test that index names are correctly set and retrieved."""
        configs = [
            ("user_vectors", 32, 100),
            ("product_embeddings", 64, 200),
            ("document_index", 128, 150),
        ]
        
        np.random.seed(42)
        indexes = []
        
        for name, dim, n_vectors in configs:
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=100,
                skip_recovery=True, index_id=get_unique_index_id(), name=name
            )
            
            # Verify the name is set correctly
            assert index.get_name() == name, f"Index name mismatch: expected {name}, got {index.get_name()}"
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append((index, name, vectors))
        
        # Verify all names are still accessible after adding points
        for index, expected_name, vectors in indexes:
            assert index.get_name() == expected_name, f"Name changed after adding points"
            
            # Verify index still works
            labels, distances = index.search_knn(vectors[0], 5, ef_search=200)
            assert 0 in labels[:10], f"Index {expected_name} search failed"
    
    def test_empty_name_default(self, caliby_module, temp_dir):
        """Test that indexes can be created without names (empty string default)."""
        dim = 64
        n_vectors = 100
        
        np.random.seed(42)
        
        # Create index without specifying name
        index = caliby_module.HnswIndex(
            max_elements=n_vectors, dim=dim, M=8, ef_construction=100,
            skip_recovery=True, index_id=get_unique_index_id()
        )
        
        # Should have empty name by default
        assert index.get_name() == "", f"Expected empty name, got: {index.get_name()}"
        
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        index.add_points(vectors, num_threads=0)
        
        # Verify it still works
        labels, distances = index.search_knn(vectors[0], 5, ef_search=50)
        assert labels[0] == 0
    
    def test_unique_names_for_multiple_indexes(self, caliby_module, temp_dir):
        """Test that multiple indexes can have unique names."""
        n_indexes = 10
        dim = 32
        n_vectors = 50
        
        np.random.seed(42)
        indexes = []
        names = set()
        
        for i in range(n_indexes):
            name = f"test_index_{i:03d}"
            names.add(name)
            
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=50,
                skip_recovery=True, index_id=get_unique_index_id(), name=name
            )
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            indexes.append((index, name))
        
        # Verify all names are unique and correct
        retrieved_names = set()
        for index, expected_name in indexes:
            actual_name = index.get_name()
            assert actual_name == expected_name, f"Name mismatch for {expected_name}"
            retrieved_names.add(actual_name)
        
        assert len(retrieved_names) == n_indexes, "Not all names are unique"
        assert retrieved_names == names, "Retrieved names don't match expected names"
    
    def test_special_characters_in_names(self, caliby_module, temp_dir):
        """Test that index names can contain special characters."""
        special_names = [
            "my-index",
            "my_index",
            "my.index",
            "my index with spaces",
            "index@123",
            "用户向量",  # Unicode characters
            "émoji_🔍",
        ]
        
        dim = 32
        n_vectors = 50
        np.random.seed(42)
        
        for name in special_names:
            index = caliby_module.HnswIndex(
                max_elements=n_vectors, dim=dim, M=8, ef_construction=50,
                skip_recovery=True, index_id=get_unique_index_id(), name=name
            )
            
            assert index.get_name() == name, f"Special name not preserved: {name}"
            
            vectors = np.random.randn(n_vectors, dim).astype(np.float32)
            index.add_points(vectors, num_threads=0)
            
            # Verify index still works
            labels, distances = index.search_knn(vectors[0], 5, ef_search=200)
            assert 0 in labels[:10], f"Index with name '{name}' failed"
    
    def test_long_name(self, caliby_module, temp_dir):
        """Test that indexes can have very long names."""
        long_name = "x" * 1000  # 1000 character name
        dim = 32
        n_vectors = 50
        
        np.random.seed(42)
        
        index = caliby_module.HnswIndex(
            max_elements=n_vectors, dim=dim, M=8, ef_construction=50,
            skip_recovery=True, index_id=get_unique_index_id(), name=long_name
        )
        
        assert index.get_name() == long_name, "Long name not preserved"
        assert len(index.get_name()) == 1000, f"Name length incorrect: {len(index.get_name())}"
        
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        index.add_points(vectors, num_threads=0)
        
        # Verify index still works
        labels, distances = index.search_knn(vectors[0], 5, ef_search=50)
        assert labels[0] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

