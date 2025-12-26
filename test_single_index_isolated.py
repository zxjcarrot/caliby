import numpy as np
import caliby

# Reproduce the exact scenario
np.random.seed(42)

_index_id_counter = 10000

# Skip first 4 indexes to get same RNG state
for idx in range(4):
    _index_id_counter += 1
    np.random.randn(10000, 32)  # Consume random numbers

# Now create index 4
_index_id_counter += 1
index_id = _index_id_counter

print(f"Creating index with id {index_id}")

index = caliby.HnswIndex(
    max_elements=10000,
    dim=32,
    M=8,
    ef_construction=50,
    skip_recovery=True,
    index_id=index_id
)

vectors = np.random.randn(10000, 32).astype(np.float32)

# Add with single thread first to see if it works
print("Adding with 1 thread...")
index.add_points(vectors, num_threads=1)

# Search
labels, distances = index.search_knn(vectors[0], 20, ef_search=200)
print(f"Single thread - Top 10: {labels[:10]}")
print(f"Found vector 0: {0 in labels[:10]}")
