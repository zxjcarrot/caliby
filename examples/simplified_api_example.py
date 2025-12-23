#!/usr/bin/env python3
"""
Caliby Simplified API Example

This example demonstrates the new, simplified Caliby API that removes
the need for config objects.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import caliby
import shutil

def main():
    data_dir = "/tmp/caliby_simple_example"
    
    # Clean up from previous runs
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    print("=" * 60)
    print("Caliby Simplified API Example")
    print("=" * 60)
    print()
    
    # Step 1: Configure buffer pool
    print("1. Configuring buffer pool...")
    caliby.set_buffer_config(size_gb=1.0)
    print("   ✓ Buffer pool configured: 1.0 GB\n")
    
    # Step 2: Open data directory
    print("2. Opening data directory...")
    caliby.open(data_dir)
    print(f"   ✓ Opened: {data_dir}\n")
    
    # Step 3: Create indexes with simplified API (no config objects!)
    print("3. Creating indexes with simplified API...")
    catalog = caliby.IndexCatalog.instance()
    
    # Create HNSW index - just pass parameters directly
    hnsw_handle = catalog.create_hnsw_index(
        name="embeddings",
        dimensions=128,
        max_elements=10000,
        M=16,                    # Optional: defaults to 16
        ef_construction=200      # Optional: defaults to 200
    )
    print(f"   ✓ Created HNSW index: {hnsw_handle.name()}")
    print(f"     - Dimensions: {hnsw_handle.dimensions()}")
    print(f"     - Max elements: {hnsw_handle.max_elements()}")
    
    # Create DiskANN index - also simplified!
    diskann_handle = catalog.create_diskann_index(
        name="vectors",
        dimensions=256,
        max_elements=50000,
        R_max_degree=64,         # Optional: defaults to 64
        L_build=100,             # Optional: defaults to 100
        alpha=1.2                # Optional: defaults to 1.2
    )
    print(f"   ✓ Created DiskANN index: {diskann_handle.name()}")
    print(f"     - Dimensions: {diskann_handle.dimensions()}")
    print(f"     - Max elements: {diskann_handle.max_elements()}")
    print()
    
    # Step 4: List all indexes
    print("4. Listing all indexes...")
    indexes = catalog.list_indexes()
    print(f"   ✓ Found {len(indexes)} indexes:")
    for info in indexes:
        index_type = "HNSW" if info.type == caliby.IndexType.HNSW else "DiskANN"
        print(f"     - {info.name} ({index_type}, {info.dimensions}D)")
    print()
    
    # Step 5: Close (flushes and releases lock automatically)
    print("5. Closing caliby...")
    caliby.close()
    print("   ✓ Closed (flushed all changes and released lock)\n")
    
    # Cleanup
    shutil.rmtree(data_dir)
    print(f"Cleaned up: {data_dir}")

if __name__ == "__main__":
    main()
