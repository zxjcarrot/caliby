import numpy as np
import caliby

np.random.seed(42)

_index_id_counter = 10000

indexes = []
vectors_list = []

# Create all 5 indexes and keep them in memory
for idx in range(5):
    _index_id_counter += 1
    index_id = _index_id_counter
    
    print(f"\nCreating index {idx} with index_id={index_id}")
    index = caliby.HnswIndex(
        max_elements=10000,
        dim=32,
        M=8,
        ef_construction=50,
        skip_recovery=True,
        index_id=index_id
    )

    vectors = np.random.randn(10000, 32).astype(np.float32)
    
    # Use 2 threads to test limited parallelism
    index.add_points(vectors, num_threads=2)
    
    indexes.append(index)
    vectors_list.append(vectors)
    print(f"Added {len(vectors)} vectors to index {idx}")

print("\n" + "="*60)
print("Verifying all indexes...")
print("="*60)

for idx, (index, vectors) in enumerate(zip(indexes, vectors_list)):
    labels, distances = index.search_knn(vectors[0], 20, ef_search=200)
    found = 0 in labels[:10]
    status = "✓ PASS" if found else "✗ FAIL"
    print(f"Index {idx}: {status} - Top 10: {labels[:10]}")
