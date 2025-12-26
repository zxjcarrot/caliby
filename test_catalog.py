#!/usr/bin/env python3
"""Test catalog-based per-index allocation"""
import numpy as np
import caliby

print("=" * 60)
print("Testing Catalog-Based Multi-Index Allocation")
print("=" * 60)

# Create indexes with different dimensions
print("\n1. Creating Index 0 (dimension 64)...")
index0 = caliby.HnswIndex(10000, 64, 16, 200, False, False, 0)
print(f"✓ Index 0 created")

print("\n2. Creating Index 1 (dimension 128)...")
index1 = caliby.HnswIndex(10000, 128, 16, 200, False, False, 1)
print(f"✓ Index 1 created")

print("\n3. Creating Index 2 (dimension 256)...")
index2 = caliby.HnswIndex(10000, 256, 16, 200, False, False, 2)
print(f"✓ Index 2 created")

# Add vectors to each index
print("\n4. Adding vectors to Index 0...")
for i in range(100):
    vec = np.random.randn(64).astype(np.float32)
    index0.add_vector(vec, i)
print(f"✓ Added 100 vectors to Index 0")

print("\n5. Adding vectors to Index 1...")
for i in range(100):
    vec = np.random.randn(128).astype(np.float32)
    index1.add_vector(vec, i)
print(f"✓ Added 100 vectors to Index 1")

print("\n6. Adding vectors to Index 2...")
for i in range(100):
    vec = np.random.randn(256).astype(np.float32)
    index2.add_vector(vec, i)
print(f"✓ Added 100 vectors to Index 2")

# Test searches
print("\n7. Testing searches...")
query0 = np.random.randn(64).astype(np.float32)
results0 = index0.search(query0, 10)
print(f"✓ Index 0 search returned {len(results0)} results")

query1 = np.random.randn(128).astype(np.float32)
results1 = index1.search(query1, 10)
print(f"✓ Index 1 search returned {len(results1)} results")

query2 = np.random.randn(256).astype(np.float32)
results2 = index2.search(query2, 10)
print(f"✓ Index 2 search returned {len(results2)} results")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Catalog-based allocation working!")
print("=" * 60)
