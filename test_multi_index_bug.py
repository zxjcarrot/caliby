import numpy as np
import caliby
import time

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
    index.add_points(vectors, num_threads=0)
    
    indexes.append(index)
    vectors_list.append(vectors)
    print(f"Added {len(vectors)} vectors to index {idx}")
    
    # Add a small delay to ensure threads have finished
    time.sleep(0.1)

print("\n" + "="*60)
print("Now searching in index 4...")
print("="*60)

# Now search in index 4
index_4 = indexes[4]
vectors_4 = vectors_list[4]

labels, distances = index_4.search_knn(vectors_4[0], 100, ef_search=500)

print(f"\nSearching for vector 0 in index 4")
print(f"Top 20 results: {labels[:20]}")

if 0 in labels[:20]:
    pos = np.where(labels == 0)[0][0]
    print(f"\n✓ Found vector 0 at position {pos}")
else:
    print(f"\n✗ Vector 0 NOT FOUND in top-20!")
