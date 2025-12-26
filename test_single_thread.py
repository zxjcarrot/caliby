#!/usr/bin/env python3
"""
Test the force_evict_buffer_portion method with single thread to isolate issue.
"""

import numpy as np
import psutil
import os
import sys
sys.path.insert(0, '/home/zxjcarrot/Workspace/caliby/build')

import caliby

process = psutil.Process()

print("="*70)
print("TESTING FORCE EVICTION (SINGLE-THREADED)")
print("="*70)

caliby.set_buffer_config(size_gb=0.3)

print("\n1. Creating index...")
index = caliby.HnswIndex(10000, 128, 16, 100, enable_prefetch=True, skip_recovery=True)

rss_baseline = process.memory_info().rss / 1024 / 1024
print(f"   RSS after index creation: {rss_baseline:.2f} MB")

print("\n2. Adding 5000 vectors (single-threaded)...")
vectors = np.random.randn(5000, 128).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
index.add_items(vectors, num_threads=1)  # FORCE SINGLE THREAD

rss_after_add = process.memory_info().rss / 1024 / 1024
print(f"   RSS after adding vectors: {rss_after_add:.2f} MB")
print(f"   RSS growth: +{rss_after_add - rss_baseline:.2f} MB")

print("\n3. Flushing to disk...")
caliby.flush_storage()

print("\n4. Forcing eviction...")
caliby.force_evict_buffer_portion(1.0)

rss_after_evict = process.memory_info().rss / 1024 / 1024
print(f"   RSS after eviction: {rss_after_evict:.2f} MB")

print("\n5. Cleanup...")
del index

print("\nSUCCESS!")
