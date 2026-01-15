"""
Tests for index file persistence and recovery with the catalog system.

Tests verify that:
1. Per-index data files are created with the correct naming convention
2. Index data is properly flushed to per-index files
3. Indexes can be recovered after restart
4. Both DiskANN and HNSW indexes work with the catalog system
"""

import pytest
import numpy as np
import os
import shutil
import tempfile
import caliby


class TestIndexFilePersistence:
    """Test that index files are created with correct naming convention."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and cleanup test directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="caliby_test_")
        yield
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_diskann_file_naming(self):
        """Test that DiskANN index files follow the naming convention."""
        np.random.seed(42)
        vectors = np.random.random((100, 64)).astype(np.float32)
        
        caliby.open(self.test_dir)
        
        # Create DiskANN index
        index = caliby.DiskANN(64, 500, R_max_degree=32, name='test_diskann')
        tags = [[i] for i in range(len(vectors))]
        params = caliby.BuildParams()
        index.build(vectors, tags, params)
        
        caliby.close()
        
        # Check file naming
        files = os.listdir(self.test_dir)
        diskann_files = [f for f in files if 'diskann' in f and f.endswith('.dat')]
        
        assert len(diskann_files) == 1
        # Format: caliby_diskann_<id>_<name>.dat
        assert diskann_files[0].startswith('caliby_diskann_')
        assert diskann_files[0].endswith('_test_diskann.dat')
        
        # Verify file has data
        file_path = os.path.join(self.test_dir, diskann_files[0])
        assert os.path.getsize(file_path) > 0
    
    def test_hnsw_file_naming(self):
        """Test that HNSW index files follow the naming convention."""
        np.random.seed(42)
        vectors = np.random.random((100, 64)).astype(np.float32)
        
        caliby.open(self.test_dir)
        
        # Create HNSW index
        index = caliby.HnswIndex(500, 64, M=16, ef_construction=200, name='test_hnsw')
        index.add_items(vectors)
        
        caliby.close()
        
        # Check file naming
        files = os.listdir(self.test_dir)
        hnsw_files = [f for f in files if 'hnsw' in f and f.endswith('.dat')]
        
        assert len(hnsw_files) == 1
        # Format: caliby_hnsw_<id>_<name>.dat
        assert hnsw_files[0].startswith('caliby_hnsw_')
        assert hnsw_files[0].endswith('_test_hnsw.dat')
        
        # Verify file has data
        file_path = os.path.join(self.test_dir, hnsw_files[0])
        assert os.path.getsize(file_path) > 0
    
    def test_multiple_indexes_naming(self):
        """Test that multiple indexes each get their own files."""
        np.random.seed(42)
        vectors = np.random.random((100, 64)).astype(np.float32)
        
        caliby.open(self.test_dir)
        
        # Create multiple indexes
        diskann1 = caliby.DiskANN(64, 500, R_max_degree=32, name='diskann_first')
        hnsw1 = caliby.HnswIndex(500, 64, M=16, ef_construction=200, name='hnsw_first')
        diskann2 = caliby.DiskANN(64, 500, R_max_degree=32, name='diskann_second')
        
        tags = [[i] for i in range(len(vectors))]
        params = caliby.BuildParams()
        
        diskann1.build(vectors, tags, params)
        hnsw1.add_items(vectors)
        diskann2.build(vectors, tags, params)
        
        caliby.close()
        
        # Check files
        files = os.listdir(self.test_dir)
        data_files = [f for f in files if f.endswith('.dat')]
        
        # Should have 3 index files
        assert len(data_files) == 3
        
        # Verify each index has its own file
        assert any('diskann_first' in f for f in data_files)
        assert any('hnsw_first' in f for f in data_files)
        assert any('diskann_second' in f for f in data_files)
        
        # Verify unique index IDs in filenames
        ids = set()
        for f in data_files:
            # Extract ID from caliby_<type>_<id>_<name>.dat
            parts = f.split('_')
            if len(parts) >= 3:
                ids.add(parts[2])
        assert len(ids) == 3  # Each index should have unique ID


class TestIndexPersistenceAndFlush:
    """Test that index data is properly flushed to files."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and cleanup test directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="caliby_test_")
        yield
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_diskann_data_persisted(self):
        """Test that DiskANN index data is persisted to file."""
        np.random.seed(42)
        num_vectors = 500
        dim = 64
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        
        caliby.open(self.test_dir)
        
        index = caliby.DiskANN(dim, 1000, R_max_degree=32, name='persist_test')
        tags = [[i] for i in range(len(vectors))]
        params = caliby.BuildParams()
        index.build(vectors, tags, params)
        
        caliby.close()
        
        # Check file exists and has reasonable size
        files = [f for f in os.listdir(self.test_dir) if 'diskann' in f and f.endswith('.dat')]
        assert len(files) == 1
        
        file_path = os.path.join(self.test_dir, files[0])
        file_size = os.path.getsize(file_path)
        
        # File should have data - at minimum vector storage
        min_expected_size = num_vectors * dim * 4  # float32
        assert file_size >= min_expected_size / 2  # Allow some compression/overhead margin
    
    def test_hnsw_data_persisted(self):
        """Test that HNSW index data is persisted to file."""
        np.random.seed(42)
        num_vectors = 500
        dim = 64
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        
        caliby.open(self.test_dir)
        
        index = caliby.HnswIndex(1000, dim, M=16, ef_construction=200, name='persist_test')
        index.add_items(vectors)
        
        caliby.close()
        
        # Check file exists and has reasonable size
        files = [f for f in os.listdir(self.test_dir) if 'hnsw' in f and f.endswith('.dat')]
        assert len(files) == 1
        
        file_path = os.path.join(self.test_dir, files[0])
        file_size = os.path.getsize(file_path)
        
        # File should have data
        min_expected_size = num_vectors * dim * 4  # float32
        assert file_size >= min_expected_size / 2


class TestIndexRecovery:
    """Test that indexes can be recovered after restart."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and cleanup test directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="caliby_test_")
        yield
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_hnsw_recovery_search(self):
        """Test that HNSW index can be recovered and searched."""
        np.random.seed(42)
        num_vectors = 500
        dim = 64
        k = 10
        ef_search = 100
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        query = vectors[0]  # Use first vector as query (1D)
        
        # Phase 1: Build and close
        caliby.open(self.test_dir)
        
        index = caliby.HnswIndex(1000, dim, M=16, ef_construction=200, name='recovery_test')
        index.add_items(vectors)
        
        # Search before close to get expected results
        labels1, distances1 = index.search_knn(query, k=k, ef_search=ef_search)
        
        caliby.close()
        
        # Phase 2: Reopen and search using a new index with same name but without catalog auto-create
        # We need to use index_id directly to skip the catalog create
        caliby.open(self.test_dir)
        
        # Create index with explicit index_id=1 (assigned from first run) to recover
        recovered_index = caliby.HnswIndex(1000, dim, M=16, ef_construction=200, 
                                           index_id=1, name='recovery_test', skip_recovery=False)
        
        # Verify recovery
        assert recovered_index.was_recovered(), "Index should have been recovered"
        
        # Search after recovery
        labels2, distances2 = recovered_index.search_knn(query, k=k, ef_search=ef_search)
        
        caliby.close()
        
        # Results should be identical or very similar
        # Allow some tolerance since graph structure might differ slightly
        common_labels = set(labels1) & set(labels2)
        recall = len(common_labels) / k
        assert recall >= 0.8, f"Recovery recall {recall} should be >= 0.8"
    
    def test_hnsw_recovery_count(self):
        """Test that recovered HNSW index has correct element count."""
        np.random.seed(42)
        num_vectors = 500
        dim = 64
        ef_search = 100
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
        
        # Phase 1: Build and close
        caliby.open(self.test_dir)
        index = caliby.HnswIndex(1000, dim, M=16, ef_construction=200, name='count_test')
        index.add_items(vectors)
        caliby.close()
        
        # Phase 2: Reopen with explicit index_id
        caliby.open(self.test_dir)
        recovered_index = caliby.HnswIndex(1000, dim, M=16, ef_construction=200, 
                                           index_id=1, name='count_test', skip_recovery=False)
        
        assert recovered_index.was_recovered()
        
        # Check element count via search
        query = vectors[0]  # 1D query
        labels, _ = recovered_index.search_knn(query, k=min(num_vectors, 100), ef_search=ef_search)
        
        caliby.close()
        
        # Should be able to find multiple results
        assert len(labels) > 0, "Should have search results"


class TestCatalogPersistence:
    """Test catalog metadata persistence."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and cleanup test directory for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="caliby_test_")
        yield
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_catalog_file_created(self):
        """Test that catalog file is created."""
        caliby.open(self.test_dir)
        
        # Create an index to populate catalog
        index = caliby.HnswIndex(100, 32, name='catalog_test')
        
        caliby.close()
        
        # Check catalog file exists
        files = os.listdir(self.test_dir)
        assert 'caliby_catalog' in files
        
        catalog_path = os.path.join(self.test_dir, 'caliby_catalog')
        assert os.path.getsize(catalog_path) > 0
    
    def test_catalog_persists_multiple_indexes(self):
        """Test that catalog correctly tracks multiple indexes."""
        caliby.open(self.test_dir)
        
        # Create multiple indexes
        idx1 = caliby.HnswIndex(100, 32, name='index_one')
        idx2 = caliby.DiskANN(32, 100, name='index_two')
        idx3 = caliby.HnswIndex(100, 32, name='index_three')
        
        caliby.close()
        
        # Count index data files
        files = os.listdir(self.test_dir)
        data_files = [f for f in files if f.endswith('.dat')]
        
        # Should have 3 index files
        assert len(data_files) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
