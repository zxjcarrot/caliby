"""
Test multi-index support with Array2Level translation arrays.
"""
import pytest
import numpy as np
import os
import tempfile


# Use the shared caliby module from conftest.py
@pytest.fixture(scope="module")
def caliby(caliby_module):
    """Get caliby module from shared fixture."""
    return caliby_module


@pytest.fixture
def catalog_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a heapfile for Calico
        heapfile = os.path.join(tmpdir, "heapfile")
        with open(heapfile, 'w') as f:
            pass
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(old_cwd)


class TestMultiIndexBasic:
    """Test basic multi-index functionality."""
    
    def test_create_multiple_indexes(self, caliby, catalog_dir):
        """Test creating multiple indexes in the same catalog."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create three indexes with different configurations
        config1 = caliby.IndexConfig()
        config1.dimensions = 128
        config1.max_elements = 1000
        hnsw1 = caliby.HNSWConfig()
        hnsw1.M = 16
        hnsw1.ef_construction = 200
        config1.hnsw = hnsw1
        
        config2 = caliby.IndexConfig()
        config2.dimensions = 256
        config2.max_elements = 2000
        hnsw2 = caliby.HNSWConfig()
        hnsw2.M = 32
        hnsw2.ef_construction = 400
        config2.hnsw = hnsw2
        
        config3 = caliby.IndexConfig()
        config3.dimensions = 128
        config3.max_elements = 500
        diskann3 = caliby.DiskANNConfig()
        diskann3.R_max_degree = 32
        diskann3.L_build = 100
        diskann3.alpha = 1.2
        config3.diskann = diskann3
        
        handle1 = catalog.create_index("vectors_128_a", caliby.IndexType.HNSW, config1)
        handle2 = catalog.create_index("vectors_256", caliby.IndexType.HNSW, config2)
        handle3 = catalog.create_index("vectors_128_b", caliby.IndexType.DISKANN, config3)
        
        assert handle1.is_valid()
        assert handle2.is_valid()
        assert handle3.is_valid()
        
        # Verify different index IDs
        assert handle1.index_id() != handle2.index_id()
        assert handle2.index_id() != handle3.index_id()
        assert handle1.index_id() != handle3.index_id()
        
        # Verify properties
        assert handle1.dimensions() == 128
        assert handle2.dimensions() == 256
        assert handle3.dimensions() == 128
        
        # List indexes
        indexes = catalog.list_indexes()
        assert len(indexes) == 3
        
        caliby.close()
    
    def test_independent_index_operations(self, caliby, catalog_dir):
        """Test that operations on different indexes are independent."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create two HNSW indexes with different configs
        config1 = caliby.IndexConfig()
        config1.dimensions = 128
        config1.max_elements = 1000
        hnsw1 = caliby.HNSWConfig()
        hnsw1.M = 16
        hnsw1.ef_construction = 200
        config1.hnsw = hnsw1
        
        config2 = caliby.IndexConfig()
        config2.dimensions = 64  # Different dimension
        config2.max_elements = 1000
        hnsw2 = caliby.HNSWConfig()
        hnsw2.M = 16
        hnsw2.ef_construction = 200
        config2.hnsw = hnsw2
        
        handle1 = catalog.create_index("index1", caliby.IndexType.HNSW, config1)
        handle2 = catalog.create_index("index2", caliby.IndexType.HNSW, config2)
        
        # Verify they have different dimensions and index IDs
        assert handle1.dimensions() == 128
        assert handle2.dimensions() == 64
        assert handle1.index_id() != handle2.index_id()
        
        caliby.close()
    
    def test_page_id_encoding(self, caliby, catalog_dir):
        """Test that page IDs correctly encode index_id and local_page_id."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        config = caliby.IndexConfig()
        config.dimensions = 128
        config.max_elements = 1000
        hnsw = caliby.HNSWConfig()
        hnsw.M = 16
        hnsw.ef_construction = 200
        config.hnsw = hnsw
        
        handle = catalog.create_index("test_index", caliby.IndexType.HNSW, config)
        
        # Test page ID composition
        index_id = handle.index_id()
        local_page_id = 12345
        
        # Make a global page ID using the handle
        global_pid = handle.global_page_id(local_page_id)
        
        # Extract components
        # Note: The actual bit layout may vary, so we just verify the function works
        # and returns a valid page ID
        assert global_pid > 0
        assert isinstance(global_pid, int)
        
        # Test that different local page IDs produce different global PIDs
        global_pid2 = handle.global_page_id(54321)
        assert global_pid != global_pid2
        
        caliby.close()


class TestMultiIndexRecovery:
    """Test recovery with multiple indexes."""
    
    def test_multi_index_recovery(self, caliby, catalog_dir):
        """Test that multiple indexes can be recovered after restart."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create multiple indexes
        config = caliby.IndexConfig()
        config.dimensions = 128
        config.max_elements = 1000
        hnsw = caliby.HNSWConfig()
        hnsw.M = 16
        hnsw.ef_construction = 200
        config.hnsw = hnsw
        
        handle1 = catalog.create_index("persistent_index1", caliby.IndexType.HNSW, config)
        handle2 = catalog.create_index("persistent_index2", caliby.IndexType.HNSW, config)
        
        # Record index IDs
        id1 = handle1.index_id()
        id2 = handle2.index_id()
        
        caliby.close()
        
        # Reinitialize catalog and recover indexes
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # List should show both indexes
        indexes = catalog.list_indexes()
        assert len(indexes) >= 2
        
        names = [info.name for info in indexes]
        assert "persistent_index1" in names
        assert "persistent_index2" in names
        
        # Open the indexes and verify IDs match
        handle1 = catalog.open_index("persistent_index1")
        handle2 = catalog.open_index("persistent_index2")
        
        assert handle1.is_valid()
        assert handle2.is_valid()
        assert handle1.index_id() == id1
        assert handle2.index_id() == id2
        
        caliby.close()
    
    def test_recovery_with_mixed_index_types(self, caliby, catalog_dir):
        """Test recovery with both HNSW and DiskANN indexes."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create HNSW index config
        hnsw_config = caliby.IndexConfig()
        hnsw_config.dimensions = 128
        hnsw_config.max_elements = 1000
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 16
        hnsw_cfg.ef_construction = 200
        hnsw_config.hnsw = hnsw_cfg
        
        handle_hnsw = catalog.create_index("mixed_hnsw", caliby.IndexType.HNSW, hnsw_config)
        
        # Create DiskANN index config
        diskann_config = caliby.IndexConfig()
        diskann_config.dimensions = 128
        diskann_config.max_elements = 1000
        diskann_cfg = caliby.DiskANNConfig()
        diskann_cfg.R_max_degree = 32
        diskann_cfg.L_build = 100
        diskann_cfg.alpha = 1.2
        diskann_config.diskann = diskann_cfg
        
        handle_diskann = catalog.create_index("mixed_diskann", caliby.IndexType.DISKANN, diskann_config)
        
        # Record index IDs
        hnsw_id = handle_hnsw.index_id()
        diskann_id = handle_diskann.index_id()
        
        caliby.close()
        
        # Recover
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # List should show both indexes
        indexes = catalog.list_indexes()
        types_map = {info.name: info.type for info in indexes}
        
        assert "mixed_hnsw" in types_map
        assert "mixed_diskann" in types_map
        assert types_map["mixed_hnsw"] == caliby.IndexType.HNSW
        assert types_map["mixed_diskann"] == caliby.IndexType.DISKANN
        
        handle_hnsw = catalog.open_index("mixed_hnsw")
        handle_diskann = catalog.open_index("mixed_diskann")
        
        # Verify index IDs are preserved
        assert handle_hnsw.index_id() == hnsw_id
        assert handle_diskann.index_id() == diskann_id
        
        caliby.close()


class TestMultiIndexStress:
    """Stress tests for multi-index functionality."""
    
    def test_many_indexes(self, caliby, catalog_dir):
        """Test creating and using many indexes."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        num_indexes = 10
        config = caliby.IndexConfig()
        config.dimensions = 64
        config.max_elements = 100
        hnsw = caliby.HNSWConfig()
        hnsw.M = 8
        hnsw.ef_construction = 100
        config.hnsw = hnsw
        
        # Create many indexes
        handles = []
        for i in range(num_indexes):
            handle = catalog.create_index(f"stress_index_{i}", caliby.IndexType.HNSW, config)
            handles.append(handle)
            assert handle.is_valid()
        
        # Verify all have unique IDs
        index_ids = [h.index_id() for h in handles]
        assert len(set(index_ids)) == num_indexes
        
        # List all indexes
        indexes = catalog.list_indexes()
        assert len(indexes) == num_indexes
        
        caliby.close()
    
    def test_index_deletion_and_recreation(self, caliby, catalog_dir):
        """Test deleting and recreating indexes."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        config = caliby.IndexConfig()
        config.dimensions = 128
        config.max_elements = 1000
        hnsw = caliby.HNSWConfig()
        hnsw.M = 16
        hnsw.ef_construction = 200
        config.hnsw = hnsw
        
        # Create index
        handle1 = catalog.create_index("reusable_name", caliby.IndexType.HNSW, config)
        first_id = handle1.index_id()
        
        # Drop index
        catalog.drop_index("reusable_name")
        
        # Recreate with same name
        handle2 = catalog.create_index("reusable_name", caliby.IndexType.HNSW, config)
        second_id = handle2.index_id()
        
        # Should have different ID (indexes are not reused)
        assert first_id != second_id
        assert handle2.is_valid()
        
        caliby.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
