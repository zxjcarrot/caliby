#!/usr/bin/env python3
import caliby
import numpy as np

print("Testing index_id parameter...")

# Test 1: Create index with index_id=0 (default, should work as before)
print("\n1. Creating index with index_id=0...")
try:
    index0 = caliby.HnswIndex(1000, 128, 16, 200, enable_prefetch=True, skip_recovery=True, index_id=0)
    print("   ✓ Index 0 created successfully")
except Exception as e:
    print(f"   ✗ Failed to create index 0: {e}")

# Test 2: Create index with index_id=1
print("\n2. Creating index with index_id=1...")
try:
    index1 = caliby.HnswIndex(1000, 128, 16, 200, enable_prefetch=True, skip_recovery=True, index_id=1)
    print("   ✓ Index 1 created successfully")
except Exception as e:
    print(f"   ✗ Failed to create index 1: {e}")

print("\nTest complete!")
