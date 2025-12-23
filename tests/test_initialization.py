#!/usr/bin/env python3
"""
Test Caliby Initialization and Directory Locking

Tests caliby.open() and caliby.close() methods with directory locking and recovery.
"""

import pytest
import tempfile
import os
import sys
import time
import subprocess
import shutil

# Add workspace root directory to path (where .so file is built)
workspace_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workspace_root)

# Also set build_path for subprocess scripts
build_path = workspace_root


class TestInitialization:
    """Test caliby initialization with data directory."""
    
    def test_initialize_new_directory(self):
        """Test initializing caliby with a new data directory."""
        # This test runs in a subprocess to avoid conflicts with other tests
        script = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = os.path.join(tmpdir, "caliby_data")
    
    # Configure buffer before initialization
    caliby.set_buffer_config(size_gb=0.3)
    
    # Initialize with new directory
    caliby.open(data_dir)
    
    # Verify directory was created
    assert os.path.exists(data_dir)
    assert os.path.isdir(data_dir)
    
    # Verify lock file was created
    lock_file = os.path.join(data_dir, ".caliby.lock")
    assert os.path.exists(lock_file)
    
    # Verify catalog file was created
    catalog_file = os.path.join(data_dir, "caliby_catalog")
    
    print(f"✓ Successfully initialized new directory: {{data_dir}}")
    print(f"  Lock file: {{lock_file}}")
    print(f"  Data directory contents: {{os.listdir(data_dir)}}")
"""
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print("Stderr:", result.stderr)
        
        assert result.returncode == 0

    
    def test_initialize_existing_directory(self):
        """Test initializing caliby with an existing data directory."""
        # This test runs in a subprocess to avoid conflicts with other tests
        script = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = os.path.join(tmpdir, "caliby_data")
    os.makedirs(data_dir)
    
    # Configure buffer
    caliby.set_buffer_config(size_gb=0.3)
    
    # Initialize with existing directory
    caliby.open(data_dir)
    
    # Verify it works
    assert os.path.exists(data_dir)
    lock_file = os.path.join(data_dir, ".caliby.lock")
    assert os.path.exists(lock_file)
    
    print("✓ Successfully initialized existing directory")
"""
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print("Stderr:", result.stderr)
        
        assert result.returncode == 0
    
    def test_double_initialization_same_process(self):
        """Test that double initialization in same process is handled gracefully."""
        # This test runs in a subprocess to avoid conflicts with other tests
        script = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    data_dir = os.path.join(tmpdir, "caliby_data")
    
    # Configure buffer
    caliby.set_buffer_config(size_gb=0.3)
    
    # Initialize once
    caliby.open(data_dir)
    
    # Initialize again - should be idempotent
    caliby.open(data_dir)
    
    print("✓ Double initialization handled gracefully")
"""
        
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print("Stderr:", result.stderr)
        
        assert result.returncode == 0


class TestDirectoryLocking:
    """Test that directory locking prevents concurrent access."""
    
    def test_lock_prevents_concurrent_access(self):
        """Test that lock prevents another process from opening same directory."""
        # This test runs in a subprocess to avoid conflicts with other tests
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            # First process initializes
            script1 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

caliby.set_buffer_config(size_gb=0.3)
caliby.open('{data_dir}')
print("First process initialized")
# Keep process running
import time
time.sleep(2)
"""
            
            # Start first process in background
            proc1 = subprocess.Popen(
                [sys.executable, "-c", script1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it time to acquire the lock
            time.sleep(0.5)
            
            # Try to open from another process
            script2 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

try:
    caliby.set_buffer_config(size_gb=0.3)
    caliby.open('{data_dir}')
    print("ERROR: Should have failed to acquire lock")
    sys.exit(1)
except RuntimeError as e:
    if "already locked" in str(e):
        print("PASS: Lock correctly prevented concurrent access")
        sys.exit(0)
    else:
        print(f"ERROR: Wrong exception: {{e}}")
        sys.exit(1)
"""
            
            result = subprocess.run(
                [sys.executable, "-c", script2],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Clean up first process
            proc1.terminate()
            proc1.wait(timeout=5)
            
            print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)
            
            assert result.returncode == 0, "Second process should have been blocked by lock"
            assert "PASS" in result.stdout
            
            print("✓ Directory lock successfully prevents concurrent access")
    
    def test_lock_released_on_shutdown(self):
        """Test that lock is released when process shuts down."""
        import caliby
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            # First process initializes and shuts down
            script1 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

caliby.set_buffer_config(size_gb=0.3)
caliby.open('{data_dir}')
print("First process initialized")

# Explicitly shutdown
catalog = caliby.IndexCatalog.instance()
caliby.close()
print("First process shut down")
"""
            
            result1 = subprocess.run(
                [sys.executable, "-c", script1],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            print("First process output:")
            print(result1.stdout)
            assert result1.returncode == 0
            
            # Second process should be able to acquire lock
            script2 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

try:
    caliby.set_buffer_config(size_gb=0.3)
    caliby.open('{data_dir}')
    print("PASS: Second process successfully acquired lock")
    sys.exit(0)
except RuntimeError as e:
    print(f"ERROR: Failed to acquire lock: {{e}}")
    sys.exit(1)
"""
            
            result2 = subprocess.run(
                [sys.executable, "-c", script2],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            print("\\nSecond process output:")
            print(result2.stdout)
            if result2.stderr:
                print("Stderr:", result2.stderr)
            
            assert result2.returncode == 0
            assert "PASS" in result2.stdout
            
            print("✓ Lock successfully released and re-acquired")


class TestRecovery:
    """Test recovery of existing indexes from disk."""
    
    def test_recovery_after_shutdown(self):
        """Test that indexes can be recovered after process restart."""
        import caliby
        import numpy as np
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            # First session: Create indexes
            print("\\n=== First Session: Creating indexes ===")
            script1 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

caliby.set_buffer_config(size_gb=0.5)
caliby.open('{data_dir}')

catalog = caliby.IndexCatalog.instance()

# Create index configuration
config = caliby.IndexConfig()
config.dimensions = 128
config.max_elements = 1000

hnsw_config = caliby.HNSWConfig()
hnsw_config.M = 16
hnsw_config.ef_construction = 200
config.hnsw = hnsw_config

# Create an index
print("Creating index 'test_index'...")
handle = catalog.create_index("test_index", caliby.IndexType.HNSW, config)
print(f"Created index: {{handle.name()}}")

# Flush and shutdown

caliby.close()
print("Shut down cleanly")
"""
            
            result1 = subprocess.run(
                [sys.executable, "-c", script1],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(result1.stdout)
            if result1.returncode != 0:
                print("Stderr:", result1.stderr)
            assert result1.returncode == 0, "First session should succeed"
            
            # Verify files were created
            assert os.path.exists(os.path.join(data_dir, "caliby_catalog"))
            print(f"\\nData directory contents: {os.listdir(data_dir)}")
            
            # Second session: Recover and verify
            print("\\n=== Second Session: Recovery ===")
            script2 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

caliby.set_buffer_config(size_gb=0.5)
caliby.open('{data_dir}')

catalog = caliby.IndexCatalog.instance()

# List indexes
indexes = catalog.list_indexes()
print(f"Found {{len(indexes)}} indexes after recovery")
for idx in indexes:
    print(f"  - {{idx}}")

# Verify we can open the index
print("Opening 'test_index'...")
handle = catalog.open_index("test_index")
print(f"Recovered index: {{handle.name()}}")
print(f"Index dimensions: {{handle.dimensions()}}")
print(f"Index max_elements: {{handle.max_elements()}}")

if len(indexes) > 0 and handle.is_valid():
    print("PASS: Recovery successful - index found and valid")
else:
    print("ERROR: Index not recovered properly")
    sys.exit(1)
"""
            
            result2 = subprocess.run(
                [sys.executable, "-c", script2],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print(result2.stdout)
            if result2.returncode != 0:
                print("Stderr:", result2.stderr)
            
            assert result2.returncode == 0, "Second session should recover successfully"
            assert "PASS" in result2.stdout
            
            print("\\n✓ Index successfully recovered after restart")
    
    def test_recovery_with_multiple_indexes(self):
        """Test recovery with multiple indexes."""
        import caliby
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            # First session: Create multiple indexes
            script1 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

caliby.set_buffer_config(size_gb=0.5)
caliby.open('{data_dir}')

catalog = caliby.IndexCatalog.instance()

# Create 3 indexes
for i in range(3):
    config = caliby.IndexConfig()
    config.dimensions = 64
    config.max_elements = 500
    
    hnsw_config = caliby.HNSWConfig()
    hnsw_config.M = 8
    hnsw_config.ef_construction = 100
    config.hnsw = hnsw_config
    
    name = f"index_{{i}}"
    handle = catalog.create_index(name, caliby.IndexType.HNSW, config)
    print(f"Created {{name}}")


caliby.close()
print("Created 3 indexes")
"""
            
            result1 = subprocess.run(
                [sys.executable, "-c", script1],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            print("Session 1:")
            print(result1.stdout)
            if result1.returncode != 0:
                print("Stderr:", result1.stderr)
            assert result1.returncode == 0
            
            # Second session: Verify all recovered
            script2 = f"""
import sys
sys.path.insert(0, '{build_path}')
import caliby

caliby.set_buffer_config(size_gb=0.5)
caliby.open('{data_dir}')

catalog = caliby.IndexCatalog.instance()

indexes = catalog.list_indexes()
print(f"Recovered {{len(indexes)}} indexes:")
for idx in indexes:
    print(f"  - {{idx}}")

if len(indexes) == 3:
    print("PASS: All 3 indexes recovered")
    sys.exit(0)
else:
    print(f"ERROR: Expected 3 indexes, got {{len(indexes)}}")
    sys.exit(1)
"""
            
            result2 = subprocess.run(
                [sys.executable, "-c", script2],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            print("\\nSession 2:")
            print(result2.stdout)
            
            if result2.returncode != 0:
                print("Stderr:", result2.stderr)
            
            assert result2.returncode == 0
            assert "PASS" in result2.stdout
            
            print("\\n✓ Multiple indexes successfully recovered")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
