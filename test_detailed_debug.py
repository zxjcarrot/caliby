import numpy as np
import caliby

# Test with a minimal case - just 2 threads and fewer vectors
np.random.seed(42)

_index_id_counter = 10000

indexes = []
vectors_list = []

# Create 5 indexes
for idx in range(5):
    _index_id_counter += 1
    index_id = _index_id_counter
    
    index = caliby.HnswIndex(
        max_elements=10000,
        dim=32,
        M=8,
        ef_construction=50,
        skip_recovery=True,
        index_id=index_id
    )

    vectors = np.random.randn(10000, 32).astype(np.float32)
    
    # Use 2 threads
    index.add_points(vectors, num_threads=2)
    
    indexes.append(index)
    vectors_list.append(vectors)

# Now verify index 4 with detailed output
index_4 = indexes[4]
vectors_4 = vectors_list[4]

# Check if vector 0 can be found at all
for k_test in [10, 50, 100, 500, 1000]:
    labels, distances = index_4.search_knn(vectors_4[0], k_test, ef_search=max(k_test * 2, 200))
    if 0 in labels:
        pos = np.where(labels == 0)[0][0]
        print(f"k={k_test}: Found vector 0 at position {pos}")
        break
else:
    print(f"Vector 0 not found even with k=1000!")
    
# Also check how close the returned results are to vector 0
print(f"\nTop 10 returned: {labels[:10]}")
print(f"Distances to vector 0:")
for i, (label, dist) in enumerate(zip(labels[:10], distances[:10])):
    # Compute actual distance to vector 0
    actual_dist = np.sum((vectors_4[label] - vectors_4[0])**2)
    print(f"  {i}: node {label}, returned_dist={dist:.4f}, actual_dist_to_vec0={actual_dist:.4f}")
