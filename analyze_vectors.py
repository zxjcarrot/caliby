import numpy as np

# Reproduce the exact random state for index 4
np.random.seed(42)

# Skip first 4 indexes worth of random numbers
for idx in range(4):
    np.random.randn(10000, 32)

# Now generate index 4's vectors
vectors = np.random.randn(10000, 32).astype(np.float32)

# Compute distances from vector 0 to all other vectors
distances = np.sum((vectors - vectors[0])**2, axis=1)

# Find the nearest neighbors of vector 0
nearest = np.argsort(distances)[:20]
print("True nearest neighbors of vector 0:")
for i, idx in enumerate(nearest):
    print(f"  {i}: node {idx}, distance={distances[idx]:.4f}")
    
# Also check what node 4605 looks like (it's the one search finds)
print(f"\nDistance from vector 0 to 4605: {distances[4605]:.4f}")
print(f"4605's rank among nearest neighbors: {np.where(np.argsort(distances) == 4605)[0][0]}")
