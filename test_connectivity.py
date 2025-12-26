import numpy as np
import caliby

# Reproduce failing index 4 with 2 threads
np.random.seed(42)
for skip in range(4):
    np.random.randn(10000, 32)

index = caliby.HnswIndex(
    max_elements=10000, dim=32, M=8, ef_construction=50,
    skip_recovery=True, index_id=50004
)

vectors = np.random.randn(10000, 32).astype(np.float32)
index.add_points(vectors, num_threads=2)

# Try to find node 0 by searching from different starting points
print("Entry point search:")
labels, _ = index.search_knn(vectors[0], 100, ef_search=500)
print(f"Found node 0: {0 in labels}")

# Check if we can find node 0 when searching for its actual neighbors
# Node 4605 is the nearest neighbor to 0
print("\nSearching for node 4605's vector:")
labels_4605, _ = index.search_knn(vectors[4605], 100, ef_search=500)
print(f"Found node 4605: {4605 in labels_4605}")
print(f"Found node 0 (should be near 4605): {0 in labels_4605}")

# Compute how close 0 should be to 4605
dist_0_to_4605 = np.sum((vectors[0] - vectors[4605])**2)
print(f"Distance from 0 to 4605: {dist_0_to_4605:.2f}")

# Check what nodes ARE returned when searching for vec[0]
print(f"\nActual results for vec[0] query: {labels[:10]}")
