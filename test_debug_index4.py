import numpy as np
import caliby

np.random.seed(42)

_index_id_counter = 10000

# Skip to index 4 which fails
for idx in range(5):
    _index_id_counter += 1
    
# Now create the problematic index 4
index_id = _index_id_counter
print(f"Creating index 4 with index_id={index_id}")

index = caliby.HnswIndex(
    max_elements=10000,
    dim=32,
    M=8,
    ef_construction=50,
    skip_recovery=True,
    index_id=index_id
)

vectors = np.random.randn(10000, 32).astype(np.float32)
index.add_points(vectors, num_threads=0)

# Search for vector 0 with very large k and ef_search
labels, distances = index.search_knn(vectors[0], 100, ef_search=500)

print(f"\nSearching for vector 0 in index 4")
print(f"Top 20 results: {labels[:20]}")

if 0 in labels:
    pos = np.where(labels == 0)[0][0]
    print(f"\n✓ Found vector 0 at position {pos}")
    print(f"Distance: {distances[pos]}")
else:
    print(f"\n✗ Vector 0 NOT FOUND in top-100 results!")
    print(f"This means vector 0 is disconnected from the graph")
    
# Also try searching from vector 0's actual location to see if others can find it
print(f"\nNow searching from vector 1 to see what it finds:")
labels2, distances2 = index.search_knn(vectors[1], 20, ef_search=200)
print(f"Top 10 when searching from vector 1: {labels2[:10]}")
