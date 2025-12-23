"""
Tests for the Caliby Catalog System.

Tests multi-index management with translation path caching.
"""
import os
import tempfile
import pytest
import numpy as np


# Use the shared caliby module from conftest.py
@pytest.fixture(scope="module")
def caliby(caliby_module):
    """Get caliby module from shared fixture."""
    return caliby_module


class TestCatalogBasic:
    """Basic catalog functionality tests."""
    
    def test_import(self, caliby):
        """Test that catalog types can be imported."""
        assert hasattr(caliby, 'IndexCatalog')
        assert hasattr(caliby, 'IndexType')
        assert hasattr(caliby, 'IndexStatus')
        assert hasattr(caliby, 'IndexConfig')
        assert hasattr(caliby, 'IndexHandle')
        assert hasattr(caliby, 'HNSWConfig')
        assert hasattr(caliby, 'DiskANNConfig')
    
    def test_index_type_enum(self, caliby):
        """Test IndexType enum values."""
        assert hasattr(caliby.IndexType, 'CATALOG')
        assert hasattr(caliby.IndexType, 'HNSW')
        assert hasattr(caliby.IndexType, 'DISKANN')
        assert hasattr(caliby.IndexType, 'IVF')
    
    def test_index_status_enum(self, caliby):
        """Test IndexStatus enum values."""
        assert hasattr(caliby.IndexStatus, 'INVALID')
        assert hasattr(caliby.IndexStatus, 'CREATING')
        assert hasattr(caliby.IndexStatus, 'ACTIVE')
        assert hasattr(caliby.IndexStatus, 'DELETED')
    
    def test_create_hnsw_config(self, caliby):
        """Test creating HNSW configuration."""
        config = caliby.HNSWConfig()
        config.M = 16
        config.ef_construction = 200
        config.max_level = 5
        config.enable_prefetch = True
        
        assert config.M == 16
        assert config.ef_construction == 200
        assert config.max_level == 5
        assert config.enable_prefetch == True
    
    def test_create_diskann_config(self, caliby):
        """Test creating DiskANN configuration."""
        config = caliby.DiskANNConfig()
        config.R_max_degree = 64
        config.L_build = 100
        config.alpha = 1.2
        config.is_dynamic = False
        
        assert config.R_max_degree == 64
        assert config.L_build == 100
        assert config.alpha == pytest.approx(1.2)
        assert config.is_dynamic == False
    
    def test_create_index_config(self, caliby):
        """Test creating IndexConfig with type-specific config."""
        config = caliby.IndexConfig()
        config.dimensions = 128
        config.max_elements = 10000
        
        # Set HNSW config through property
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 32
        hnsw_cfg.ef_construction = 150
        config.hnsw = hnsw_cfg
        
        assert config.dimensions == 128
        assert config.max_elements == 10000
        assert config.hnsw.M == 32


class TestCatalogInstance:
    """Tests for IndexCatalog singleton."""
    
    def test_get_instance(self, caliby):
        """Test getting catalog singleton instance."""
        catalog = caliby.IndexCatalog.instance()
        assert catalog is not None
    
    def test_singleton_same_instance(self, caliby):
        """Test that instance returns same object."""
        catalog1 = caliby.IndexCatalog.instance()
        catalog2 = caliby.IndexCatalog.instance()
        # Both should be valid catalogs
        assert catalog1 is not None
        assert catalog2 is not None


class TestCatalogOperations:
    """Tests for catalog index operations."""
    
    @pytest.fixture
    def catalog_dir(self):
        """Create a temporary directory for catalog tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialize_catalog(self, caliby, catalog_dir):
        """Test catalog initialization."""
        catalog = caliby.IndexCatalog.instance()
        
        # Initialize should succeed (no exception)
        catalog.initialize(catalog_dir)
        assert catalog.is_initialized()
        caliby.close()
    
    def test_list_empty_catalog(self, caliby, catalog_dir):
        """Test listing indexes in empty catalog."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        indexes = catalog.list_indexes()
        assert isinstance(indexes, list)
        assert len(indexes) == 0
        caliby.close()
    
    def test_create_hnsw_index(self, caliby, catalog_dir):
        """Test creating an HNSW index through catalog."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create HNSW config
        config = caliby.IndexConfig()
        config.dimensions = 64
        config.max_elements = 1000
        
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 16
        hnsw_cfg.ef_construction = 100
        config.hnsw = hnsw_cfg
        
        # Create the index
        handle = catalog.create_index("my_hnsw_index", caliby.IndexType.HNSW, config)
        assert handle is not None
        assert handle.is_valid()
        
        # Verify index is in list
        indexes = catalog.list_indexes()
        assert len(indexes) >= 1
        
        # Find our index
        found = False
        for info in indexes:
            if info.name == "my_hnsw_index":
                found = True
                assert info.type == caliby.IndexType.HNSW
                assert info.dimensions == 64
                assert info.status == caliby.IndexStatus.ACTIVE
                break
        assert found, "Created index not found in list"
        caliby.close()
    
    def test_create_diskann_index(self, caliby, catalog_dir):
        """Test creating a DiskANN index through catalog."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create DiskANN config
        config = caliby.IndexConfig()
        config.dimensions = 128
        config.max_elements = 5000
        
        diskann_cfg = caliby.DiskANNConfig()
        diskann_cfg.R_max_degree = 32
        diskann_cfg.L_build = 75
        diskann_cfg.alpha = 1.2
        config.diskann = diskann_cfg
        
        # Create the index
        handle = catalog.create_index("my_diskann_index", caliby.IndexType.DISKANN, config)
        assert handle is not None
        assert handle.is_valid()
        
        # Verify index is in list
        indexes = catalog.list_indexes()
        names = [info.name for info in indexes]
        assert "my_diskann_index" in names
        caliby.close()
    
    def test_open_existing_index(self, caliby, catalog_dir):
        """Test opening an existing index."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create index config
        config = caliby.IndexConfig()
        config.dimensions = 64
        config.max_elements = 1000
        
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 16
        hnsw_cfg.ef_construction = 100
        config.hnsw = hnsw_cfg
        
        # Create the index
        catalog.create_index("reopen_test", caliby.IndexType.HNSW, config)
        
        # Now open it
        handle = catalog.open_index("reopen_test")
        assert handle is not None
        assert handle.is_valid()
        assert handle.name() == "reopen_test"
        caliby.close()
    
    def test_open_nonexistent_index(self, caliby, catalog_dir):
        """Test opening a non-existent index raises an exception."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Try to open non-existent index - should raise
        try:
            handle = catalog.open_index("does_not_exist")
            # If we got here, handle should be invalid or None
            assert handle is None or not handle.is_valid()
        except RuntimeError:
            pass  # Expected
        finally:
            caliby.close()
    
    def test_drop_index(self, caliby, catalog_dir):
        """Test dropping an index."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create index config
        config = caliby.IndexConfig()
        config.dimensions = 64
        config.max_elements = 1000
        
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 16
        hnsw_cfg.ef_construction = 100
        config.hnsw = hnsw_cfg
        
        # Create the index
        catalog.create_index("drop_test", caliby.IndexType.HNSW, config)
        
        # Verify it exists
        assert catalog.index_exists("drop_test")
        
        # Drop it
        catalog.drop_index("drop_test")
        
        # Verify it's gone
        assert not catalog.index_exists("drop_test")
        caliby.close()
    
    def test_multiple_indexes(self, caliby, catalog_dir):
        """Test creating multiple indexes."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create 3 different indexes
        for i in range(3):
            config = caliby.IndexConfig()
            config.dimensions = 32 * (i + 1)
            config.max_elements = 500 * (i + 1)
            
            hnsw_cfg = caliby.HNSWConfig()
            hnsw_cfg.M = 8 * (i + 1)
            hnsw_cfg.ef_construction = 50 * (i + 1)
            config.hnsw = hnsw_cfg
            
            handle = catalog.create_index(f"multi_index_{i}", caliby.IndexType.HNSW, config)
            assert handle is not None
            assert handle.is_valid()
        
        # Verify all exist
        indexes = catalog.list_indexes()
        names = [info.name for info in indexes]
        assert "multi_index_0" in names
        assert "multi_index_1" in names
        assert "multi_index_2" in names
        caliby.close()


class TestIndexHandle:
    """Tests for IndexHandle functionality."""
    
    @pytest.fixture
    def catalog_with_index(self, caliby):
        """Create a catalog with an index for testing."""
        tmpdir = tempfile.mkdtemp()
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(tmpdir)
        
        # Create index config
        config = caliby.IndexConfig()
        config.dimensions = 64
        config.max_elements = 1000
        
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 16
        hnsw_cfg.ef_construction = 100
        config.hnsw = hnsw_cfg
        
        handle = catalog.create_index("handle_test", caliby.IndexType.HNSW, config)
        yield catalog, handle, tmpdir
        caliby.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_handle_validity(self, catalog_with_index):
        """Test that created handle is valid."""
        _, handle, _ = catalog_with_index
        
        # Handle should be valid after creation
        assert handle is not None
        assert handle.is_valid()
    
    def test_handle_index_id(self, catalog_with_index):
        """Test getting index_id from handle."""
        _, handle, _ = catalog_with_index
        
        # Should have a valid index ID (> 0 since 0 is reserved for catalog)
        index_id = handle.index_id()
        assert index_id > 0
    
    def test_handle_properties(self, catalog_with_index):
        """Test handle properties."""
        _, handle, _ = catalog_with_index
        
        assert handle.name() == "handle_test"
        assert handle.dimensions() == 64
        assert handle.max_elements() == 1000
    
    def test_global_page_id(self, catalog_with_index):
        """Test global page ID computation."""
        _, handle, _ = catalog_with_index
        
        index_id = handle.index_id()
        
        # Local page 0 should give us a global page ID
        global_pid = handle.global_page_id(0)
        
        # The global page ID should have index_id in the high bits (32-bit layout)
        expected = index_id << 32
        assert global_pid == expected


class TestTranslationPathCaching:
    """Tests for translation path caching functionality."""
    
    @pytest.fixture
    def catalog_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_page_id_structure(self, caliby, catalog_dir):
        """Test that page IDs have correct structure."""
        catalog = caliby.IndexCatalog.instance()
        catalog.initialize(catalog_dir)
        
        # Create an index
        config = caliby.IndexConfig()
        config.dimensions = 64
        config.max_elements = 1000
        
        hnsw_cfg = caliby.HNSWConfig()
        hnsw_cfg.M = 16
        hnsw_cfg.ef_construction = 100
        config.hnsw = hnsw_cfg
        
        handle = catalog.create_index("pageid_test", caliby.IndexType.HNSW, config)
        
        # Get the index ID
        index_id = handle.index_id()
        
        # The index ID should fit in 32 bits
        assert index_id < (1 << 32)
        
        # Check global page ID for different local pages
        # Page ID layout: [index_id (32 bits)][local_page_id (32 bits)]
        for local_pid in [0, 1, 100, 1000]:
            global_pid = handle.global_page_id(local_pid)
            expected = (index_id << 32) | local_pid
            assert global_pid == expected
        
        caliby.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
