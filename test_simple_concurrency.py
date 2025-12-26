import numpy as np
import caliby

np.random.seed(42)

_index_id_counter = 10000

# Create 20 small indexes like the failing test
for idx in range(20):
    _index_id_counter += 1
    index_id = _index_id_counter
    
    print(f"\n=== Creating index {idx} with index_id={index_id} ===")
    index = caliby.HnswIndex(
        max_elements=10000,
        dim=32,
        M=8,
        ef_construction=50,
        skip_recovery=True,
        index_id=index_id
    )

    # Add vectors in parallel
    vectors = np.random.randn(10000, 32).astype(np.float32)
    index.add_points(vectors, num_threads=0)  # 0 means use all threads

    # Search for the first vector
    labels, distances = index.search_knn(vectors[0], 20, ef_search=200)

    print(f"Searched for vector 0")
    print(f"Top 10 results: {labels[:10]}")

    if 0 not in labels[:10]:
        print(f"\nERROR: Index {idx} failed to find vector 0 in top 10!")
        print(f"All results: {labels}")
        break
    else:
        print(f"SUCCESS: Index {idx} found vector 0!")
