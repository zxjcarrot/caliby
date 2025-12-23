#!/usr/bin/env python3
"""
Example: Building and querying an HNSW index with Caliby

This example demonstrates:
1. Creating an HNSW index with a buffer pool
2. Adding vectors to the index
3. Performing k-NN search
4. Checking index statistics
"""

import numpy as np
import caliby
import tempfile
import os



# Configure buffer pool sizes
caliby.set_buffer_config(size_gb=0.3)
def main():
    # Configuration
    dim = 128          # Vector dimension
    num_vectors = 10000  # Number of vectors to index
    k = 10             # Number of nearest neighbors to find
    
    # HNSW parameters
    M = 16             # Number of connections per layer
    ef_construction = 200  # Size of dynamic candidate list during construction
    ef_search = 100    # Size of dynamic candidate list during search
    
    # Create temporary directory for index storage
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "hnsw_index")
        
        print(f"Creating HNSW index at: {index_path}")
        print(f"  Dimension: {dim}")
        print(f"  M: {M}, ef_construction: {ef_construction}")
        
        # Create HNSW index
        # Parameters: path, dim, M, ef_construction
        index = caliby.HNSWIndex(index_path, dim, M, ef_construction)
        
        # Generate random vectors
        print(f"\nGenerating {num_vectors} random vectors...")
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        
        # Normalize vectors (optional, for cosine similarity)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Add vectors to the index
        print("Adding vectors to index...")
        for i, vec in enumerate(vectors):
            index.add_point(vec, i)
            if (i + 1) % 1000 == 0:
                print(f"  Added {i + 1}/{num_vectors} vectors")
        
        print(f"\nIndex built successfully!")
        
        # Set search parameter
        index.set_ef(ef_search)
        
        # Create a query vector (use the first vector for exact match test)
        query = vectors[0].copy()
        
        print(f"\nSearching for {k} nearest neighbors...")
        
        # Search returns (distances, labels) tuples
        distances, labels = index.search_knn(query, k)
        
        print("\nSearch results:")
        print(f"  Query vector index: 0")
        print(f"  Top {k} nearest neighbors:")
        for i, (dist, label) in enumerate(zip(distances, labels)):
            print(f"    {i+1}. Label: {label}, Distance: {dist:.6f}")
        
        # Verify that the closest neighbor is the query itself
        if labels[0] == 0:
            print("\n✓ Correct! The closest vector is the query itself.")
        else:
            print(f"\n✗ Unexpected: closest neighbor is {labels[0]}, expected 0")
        
        # Test with a random query
        print("\n" + "="*50)
        print("Testing with a random query vector...")
        random_query = np.random.randn(dim).astype(np.float32)
        random_query = random_query / np.linalg.norm(random_query)
        
        distances, labels = index.search_knn(random_query, k)
        
        print(f"Top {k} nearest neighbors for random query:")
        for i, (dist, label) in enumerate(zip(distances, labels)):
            print(f"  {i+1}. Label: {label}, Distance: {dist:.6f}")


if __name__ == "__main__":
    main()
