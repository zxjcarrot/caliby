#!/usr/bin/env python3
"""
Test the force_evict_buffer_portion method to verify hole punching works.
"""

import numpy as np
import psutil
import os


def test_force_eviction(caliby_module):
    """Test force eviction with hole punching (config set in conftest.py)."""
    process = psutil.Process()
    
    print("="*70)
    print("TESTING FORCE EVICTION AND HOLE PUNCHING")
    print("="*70)
    
    print("\n1. Creating index...")
    index = caliby_module.HnswIndex(10000, 128, 16, 100, enable_prefetch=True, skip_recovery=True)
    
    rss_baseline = process.memory_info().rss / 1024 / 1024
    print(f"   RSS after index creation: {rss_baseline:.2f} MB")
    
    print("\n2. Adding 5000 vectors...")
    vectors = np.random.randn(5000, 128).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add_items(vectors)
    
    rss_after_add = process.memory_info().rss / 1024 / 1024
    print(f"   RSS after adding vectors: {rss_after_add:.2f} MB")
    print(f"   RSS growth: +{rss_after_add - rss_baseline:.2f} MB")
    
    print("\n3. Flushing to disk to unlock all pages...")
    caliby_module.flush_storage()
    print(f"   Flushed all dirty pages")
    
    rss_after_flush = process.memory_info().rss / 1024 / 1024
    print(f"   RSS after flush: {rss_after_flush:.2f} MB")
    
    print("\n4. Forcing eviction of 100% of buffer pool (triggers hole punching)...")
    caliby_module.force_evict_buffer_portion(1.0)
    
    rss_after_evict = process.memory_info().rss / 1024 / 1024
    print(f"   RSS after force eviction: {rss_after_evict:.2f} MB")
    print(f"   RSS freed: {rss_after_flush - rss_after_evict:.2f} MB")
    
    print("\n5. Cleanup...")
    del index
    
    print("\n" + "="*70)
    print("CHECK HOLE PUNCH COUNTER IN OUTPUT BELOW:")
    print("="*70)

if __name__ == "__main__":
    test_force_eviction()
