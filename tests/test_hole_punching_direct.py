#!/usr/bin/env python3
"""
Direct test of hole punching by forcing small buffer pool.

This script tests hole punching by using caliby.set_buffer_config()
to set a small buffer pool, forcing evictions.
"""

import os
import sys
import time

# Add build directory to path
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

import numpy as np
import psutil


def get_rss_mb():
    """Get current RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    print("="*70)
    print("Testing Hole Punching with Small Buffer Pool ()")
    print("="*70)
    
    # Clean up old files
    for f in ['catalog.dat', 'heapfile']:
        if os.path.exists(f):
            os.remove(f)
    
    import caliby
    
    # Configure buffer pool sizes
    caliby.set_buffer_config(size_gb=1)
    baseline_rss = get_rss_mb()
    print(f"\n1. Baseline RSS: {baseline_rss:.2f} MB")
    print(f"   Buffer pool size: 1GB (will force evictions)")
    
    # Create multiple large indexes that exceed 1GB buffer
    dim = 256
    n_vectors = 5000
    n_indexes = 8
    
    print(f"\n2. Creating {n_indexes} indexes with {n_vectors} vectors each...")
    print(f"   This should exceed the 1GB buffer and force evictions")
    
    indexes = []
    np.random.seed(42)
    
    for idx in range(n_indexes):
        print(f"\n   Index {idx}:")
        index = caliby.HnswIndex(
            max_elements=n_vectors,
            dim=dim,
            M=16,
            ef_construction=100,
            skip_recovery=True,
            index_id=idx + 1,
            name=f"index_{idx}"
        )
        
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        print(f"      Added {n_vectors} vectors")
        
        # Perform searches to make pages resident
        for i in range(200):
            index.search_knn(vectors[i % len(vectors)], k=10, ef_search=50)
        print(f"      Completed 200 searches")
        
        indexes.append((index, vectors))
        
        current_rss = get_rss_mb()
        print(f"      Current RSS: {current_rss:.2f} MB (+{current_rss - baseline_rss:.2f} MB)")
    
    peak_rss = get_rss_mb()
    print(f"\n3. Peak RSS: {peak_rss:.2f} MB")
    print(f"   Memory used: {peak_rss - baseline_rss:.2f} MB")
    
    # Now perform intensive cross-index operations to force evictions
    print(f"\n4. Performing intensive cross-index operations...")
    print(f"   This should cause buffer pool to evict and trigger hole punching")
    
    for round_num in range(5):
        # Random access pattern across all indexes
        for _ in range(100):
            idx = np.random.randint(0, len(indexes))
            index, vectors = indexes[idx]
            query_idx = np.random.randint(0, len(vectors))
            index.search_knn(vectors[query_idx], k=10, ef_search=50)
        
        current_rss = get_rss_mb()
        print(f"   Round {round_num + 1}: RSS = {current_rss:.2f} MB")
    
    after_ops_rss = get_rss_mb()
    print(f"\n5. RSS after operations: {after_ops_rss:.2f} MB")
    
    # The key metric: RSS shouldn't grow linearly with operations
    # If hole punching works, RSS should stay bounded
    rss_growth = after_ops_rss - peak_rss
    print(f"   RSS growth during operations: {rss_growth:+.2f} MB")
    
    if abs(rss_growth) < 100:
        print(f"   ✓ RSS remained bounded (hole punching likely working)")
    else:
        print(f"   ⚠ RSS grew significantly")
    
    # Cleanup
    print(f"\n6. Cleaning up...")
    for index, _ in indexes:
        del index
    del indexes
    
    import gc
    gc.collect()
    time.sleep(0.5)
    
    final_rss = get_rss_mb()
    print(f"\n7. Final RSS: {final_rss:.2f} MB")
    print(f"   Memory freed: {peak_rss - final_rss:.2f} MB")
    
    print("\n" + "="*70)
    print("IMPORTANT: Check the output above for this line:")
    print("[BufferManager] Total hole punches (madvise count): X")
    print("")
    print("If X > 0, hole punching is working!")
    print("If X = 0, the buffer was large enough to hold all data")
    print("="*70)


if __name__ == "__main__":
    main()
