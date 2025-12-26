#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import caliby

print("Testing TRADHASH=0 (Array1Access mode)...")
# BufferManager is created automatically, just create an index
idx = caliby.HnswIndex(1000, 128, 16, 200)
print("✓ Initialization successful")

# Test basic operations
import numpy as np
test_vector = np.random.rand(128).astype(np.float32)
idx.add_items(test_vector.reshape(1, -1), np.array([0]))
print("✓ Add item successful")

result_labels, result_distances = idx.search_knn(test_vector, k=1)
print(f"✓ Search successful: found label {result_labels[0]}")

print("\nTRADHASH=0 test passed!")
