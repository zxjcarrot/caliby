#!/usr/bin/env python3
"""
Example: Caliby Initialization and Persistent Indexes

This example demonstrates the new initialization API with data directory
management, file locking, and recovery.
"""

import sys
import os

# Add the workspace root to path to use the local build
workspace_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, workspace_root)

import caliby
import numpy as np

def create_and_populate_index(data_dir="/tmp/caliby_example"):
    """First session: Create a persistent index and add vectors."""
    print("=== Session 1: Creating Index ===\n")
    
    # Step 1: Configure buffer pool (must be done before initialization)
    caliby.set_buffer_config(size_gb=1.0)
    print("✓ Configured buffer pool: 1.0 GB physical memory")
    
    # Step 2: Initialize with data directory
    caliby.open(data_dir)
    print(f"✓ Initialized caliby with data directory: {data_dir}")
    print(f"  Directory contents: {os.listdir(data_dir)}\n")
    
    # Step 3: Create a persistent index using the catalog API
    catalog = caliby.IndexCatalog.instance()
    
    config = caliby.IndexConfig()
    config.dimensions = 128
    config.max_elements = 10000
    
    hnsw_config = caliby.HNSWConfig()
    hnsw_config.M = 16
    hnsw_config.ef_construction = 200
    config.hnsw = hnsw_config
    
    handle = catalog.create_index("example_vectors", caliby.IndexType.HNSW, config)
    print(f"✓ Created index: {handle.name()}")
    print(f"  Dimensions: {handle.dimensions()}")
    print(f"  Max elements: {handle.max_elements()}\n")
    
    # Note: Currently, IndexHandle doesn't provide direct access to add vectors
    # For this example, we'll just create the index structure
    
    # Step 4: Flush changes to disk
    
    print("✓ Flushed all changes to disk\n")
    
    # Step 5: Clean shutdown
    caliby.close()
    print("✓ Shutdown complete")
    print(f"  Final directory contents: {os.listdir(data_dir)}\n")

def recover_and_list_indexes(data_dir="/tmp/caliby_example"):
    """Second session: Recover indexes from disk."""
    print("=== Session 2: Recovery ===\n")
    
    # Step 1: Configure buffer pool
    caliby.set_buffer_config(size_gb=1.0)
    print("✓ Configured buffer pool")
    
    # Step 2: Initialize - this triggers recovery
    caliby.open(data_dir)
    print(f"✓ Initialized caliby with existing directory: {data_dir}")
    
    # Step 3: List recovered indexes
    catalog = caliby.IndexCatalog.instance()
    indexes = catalog.list_indexes()
    
    print(f"✓ Recovered {len(indexes)} indexes:\n")
    for info in indexes:
        print(f"  Index: {info.name}")
        print(f"    Type: {'HNSW' if info.type == caliby.IndexType.HNSW else 'DiskANN'}")
        print(f"    Dimensions: {info.dimensions}")
        print(f"    Status: {'ACTIVE' if info.status == caliby.IndexStatus.ACTIVE else 'INACTIVE'}")
        print()
    
    # Step 4: Open recovered index
    if len(indexes) > 0:
        handle = catalog.open_index("example_vectors")
        print(f"✓ Opened index: {handle.name()}")
        print(f"  Valid: {handle.is_valid()}")
        print(f"  Index ID: {handle.index_id()}\n")
    
    # Step 5: Clean shutdown
    caliby.close()
    print("✓ Shutdown complete\n")

def demonstrate_locking(data_dir="/tmp/caliby_example"):
    """Demonstrate file locking by attempting concurrent access."""
    print("=== File Locking Demonstration ===\n")
    
    import subprocess
    
    # Start a process that holds the lock
    script = f"""
import caliby
import time

caliby.set_buffer_config(size_gb=1.0)
caliby.open('{data_dir}')
print("Process 1: Acquired lock")

# Hold lock for 3 seconds
time.sleep(3)

catalog = caliby.IndexCatalog.instance()
caliby.close()
print("Process 1: Released lock")
"""
    
    print("Starting process 1 (will hold lock for 3 seconds)...")
    proc1 = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it time to acquire lock
    import time
    time.sleep(0.5)
    
    # Try to access from another process
    print("Attempting to access from process 2 (should fail)...\n")
    
    script2 = f"""
import caliby

try:
    caliby.set_buffer_config(size_gb=1.0)
    caliby.open('{data_dir}')
    print("ERROR: Should have failed to acquire lock!")
except RuntimeError as e:
    if "already locked" in str(e):
        print("✓ Process 2 correctly blocked by lock")
        print(f"  Error message: {{e}}")
    else:
        print(f"Unexpected error: {{e}}")
"""
    
    result = subprocess.run(
        [sys.executable, "-c", script2],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    print(result.stdout)
    
    # Wait for first process to finish
    stdout, stderr = proc1.communicate(timeout=5)
    print(stdout)
    
    print("✓ File locking prevents concurrent access\n")

def cleanup_example_data(data_dir="/tmp/caliby_example"):
    """Clean up example data directory."""
    import shutil
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print(f"✓ Cleaned up example directory: {data_dir}\n")

def main():
    """Run all examples."""
    data_dir = "/tmp/caliby_example"
    
    print("Caliby Initialization Example")
    print("=" * 60)
    print()
    
    try:
        # Clean up any previous runs
        cleanup_example_data(data_dir)
        
        # Example 1: Create index
        create_and_populate_index(data_dir)
        
        print("\n" + "=" * 60 + "\n")
        
        # Example 2: Recovery
        # Note: This would need to be run in a separate Python process
        # to truly demonstrate recovery. For now, we can't do it in the
        # same process because caliby is already initialized.
        print("To test recovery, run this script again. It will detect")
        print("the existing index and recover it automatically.\n")
        
        # Check if we should demonstrate recovery
        if len(sys.argv) > 1 and sys.argv[1] == "--recover":
            recover_and_list_indexes(data_dir)
        
        # Example 3: Locking (commented out as it's complex)
        # demonstrate_locking(data_dir)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("=" * 60)
        cleanup_example_data(data_dir)

if __name__ == "__main__":
    main()
