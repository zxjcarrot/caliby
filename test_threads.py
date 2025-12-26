#!/usr/bin/env python3
"""
Test script to verify that add_points() now supports num_threads parameter.
"""

import numpy as np
import caliby

# Configure buffer
caliby.set_buffer_config(virtgb=1, size_gb=1)

# Create a small test index
dim = 128
max_elements = 10000
M = 16
ef_construction = 200

print("Creating HNSW index...")
index = caliby.HnswIndex(max_elements, dim, M, ef_construction, enable_prefetch=True, skip_recovery=True)

# Generate random data
num_vectors = 5000
vectors = np.random.rand(num_vectors, dim).astype(np.float32)
print(f"Generated {num_vectors} random {dim}-dimensional vectors")

# Test 1: Default (automatic thread count)
print("\nTest 1: Adding points with default threading (num_threads=0)")
index.add_points(vectors, num_threads=0)
print(f"✓ Successfully added {num_vectors} points with automatic threading")

# Create another index for testing explicit thread count
print("\nCreating second index for explicit thread count test...")
index2 = caliby.HnswIndex(max_elements, dim, M, ef_construction, enable_prefetch=True, skip_recovery=True)

# Test 2: Explicit thread count
print("\nTest 2: Adding points with explicit thread count (num_threads=4)")
index2.add_points(vectors, num_threads=4)
print(f"✓ Successfully added {num_vectors} points with 4 threads")

# Create third index for single-threaded test
print("\nCreating third index for single-threaded test...")
index3 = caliby.HnswIndex(max_elements, dim, M, ef_construction, enable_prefetch=True, skip_recovery=True)

# Test 3: Single thread
print("\nTest 3: Adding points with single thread (num_threads=1)")
index3.add_points(vectors, num_threads=1)
print(f"✓ Successfully added {num_vectors} points with 1 thread")

print("\n✅ All tests passed! The num_threads parameter works correctly.")
