#!/usr/bin/env python3
"""
Comprehensive Hole Punching Analysis

This test documents the hole punching mechanism in Caliby and explains
when it actually triggers.
"""

import os
import sys

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

import numpy as np
import psutil
import time


def get_rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    print("="*80)
    print("HOLE PUNCHING ANALYSIS FOR CALIBY")
    print("="*80)
    
    # Clean up
    for f in ['catalog.dat', 'heapfile']:
        if os.path.exists(f):
            os.remove(f)
    
    import caliby
    

# Configure buffer pool sizes
caliby.set_buffer_config(size_gb=0.1)
    print("\n" + "="*80)
    print("UNDERSTANDING HOLE PUNCHING")
    print("="*80)
    print("""
Hole punching in Caliby uses madvise(MADV_DONTNEED) to release physical memory
back to the OS while keeping virtual memory mappings intact.

KEY INSIGHT: Hole punching only occurs when pages are EVICTED from the buffer pool!

Eviction happens when:
1. Buffer pool reaches 95% capacity (physUsedCount >= physCount * 0.95)
2. Clock algorithm selects pages to evict
3. Evicted pages trigger decrementRefCount() which calls madvise()

Buffer Pool Configuration:
- Size controlled by PHYSGB environment variable
- Default: 4GB (can be changed via PHYSGB env var)
- Page size: 4KB
- Eviction threshold: 95% of buffer capacity

Current Test Configuration:
- (100MB buffer pool)
- This should force evictions and hole punching
""")
    
    baseline_rss = get_rss_mb()
    print(f"\n1. Baseline RSS: {baseline_rss:.2f} MB")
    print(f"   Buffer Pool: 100MB (will force evictions)")
    
    # Create indexes that will exceed the tiny buffer
    dim = 128
    n_vectors = 3000
    n_indexes = 10
    
    print(f"\n2. Creating {n_indexes} indexes to exceed buffer capacity...")
    
    indexes = []
    np.random.seed(42)
    
    for idx in range(n_indexes):
        index = caliby.HnswIndex(
            max_elements=n_vectors,
            dim=dim,
            M=16,
            ef_construction=100,
            skip_recovery=True,
            index_id=idx + 1,
            name=f"test_index_{idx}"
        )
        
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        index.add_points(vectors)
        
        # Many searches to load pages
        for i in range(300):
            index.search_knn(vectors[i % len(vectors)], k=10, ef_search=50)
        
        indexes.append((index, vectors))
        
        if (idx + 1) % 3 == 0:
            print(f"   Created {idx + 1} indexes: RSS = {get_rss_mb():.2f} MB")
    
    peak_rss = get_rss_mb()
    print(f"\n3. Peak RSS: {peak_rss:.2f} MB")
    
    # Intensive operations to force evictions
    print(f"\n4. Performing intensive operations to force evictions...")
    for round_num in range(10):
        for _ in range(200):
            idx = np.random.randint(0, len(indexes))
            index, vectors = indexes[idx]
            query_idx = np.random.randint(0, len(vectors))
            index.search_knn(vectors[query_idx], k=10, ef_search=50)
        
        if (round_num + 1) % 3 == 0:
            print(f"   Round {round_num + 1}: RSS = {get_rss_mb():.2f} MB")
    
    final_ops_rss = get_rss_mb()
    print(f"\n5. RSS after intensive operations: {final_ops_rss:.2f} MB")
    print(f"   RSS growth: {final_ops_rss - peak_rss:+.2f} MB")
    
    # Cleanup
    for index, _ in indexes:
        del index
    del indexes
    import gc
    gc.collect()
    time.sleep(0.5)
    
    cleanup_rss = get_rss_mb()
    print(f"\n6. RSS after cleanup: {cleanup_rss:.2f} MB")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Baseline RSS:      {baseline_rss:.2f} MB")
    print(f"Peak RSS:          {peak_rss:.2f} MB")
    print(f"After operations:  {final_ops_rss:.2f} MB")
    print(f"After cleanup:     {cleanup_rss:.2f} MB")
    print(f"Total freed:       {peak_rss - cleanup_rss:.2f} MB")
    
    print("\n" + "="*80)
    print("CHECK HOLE PUNCH COUNTER BELOW:")
    print("="*80)


if __name__ == "__main__":
    main()
