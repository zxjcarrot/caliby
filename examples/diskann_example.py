#!/usr/bin/env python3
"""
Example: Building and querying a DiskANN index with Caliby

This example demonstrates:
1. Creating a DiskANN index optimized for SSD storage
2. Adding vectors to the index
3. Performing k-NN search with beam search
4. Understanding DiskANN's disk-optimized design
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
    
    # DiskANN parameters
    R = 64             # Maximum out-degree (graph connectivity)
    L = 100            # Beam width for search
    alpha = 1.2        # Pruning parameter (affects graph quality)
    
    # Create temporary directory for index storage
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "diskann_index")
        
        print(f"Creating DiskANN index at: {index_path}")
        print(f"  Dimension: {dim}")
        print(f"  R (max degree): {R}")
        print(f"  Alpha: {alpha}")
        
        # Create DiskANN index
        # Parameters: path, dim, R, alpha
        index = caliby.DiskANNIndex(index_path, dim, R, alpha)
        
        # Generate random vectors
        print(f"\nGenerating {num_vectors} random vectors...")
        np.random.seed(42)
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        
        # Normalize vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Add vectors to the index
        print("Adding vectors to index...")
        for i, vec in enumerate(vectors):
            index.add_point(vec, i)
            if (i + 1) % 1000 == 0:
                print(f"  Added {i + 1}/{num_vectors} vectors")
        
        print(f"\nIndex built successfully!")
        
        # Create a query vector
        query = vectors[0].copy()
        
        print(f"\nSearching for {k} nearest neighbors with beam width L={L}...")
        
        # Search with beam width L
        distances, labels = index.search_knn(query, k, L)
        
        print("\nSearch results:")
        print(f"  Query vector index: 0")
        print(f"  Top {k} nearest neighbors:")
        for i, (dist, label) in enumerate(zip(distances, labels)):
            print(f"    {i+1}. Label: {label}, Distance: {dist:.6f}")
        
        # Verify result
        if labels[0] == 0:
            print("\n✓ Correct! The closest vector is the query itself.")
        else:
            print(f"\n✗ Unexpected: closest neighbor is {labels[0]}, expected 0")
        
        # Demonstrate effect of beam width on recall
        print("\n" + "="*50)
        print("Effect of beam width (L) on search quality:")
        
        random_query = np.random.randn(dim).astype(np.float32)
        random_query = random_query / np.linalg.norm(random_query)
        
        for beam_width in [10, 50, 100, 200]:
            distances, labels = index.search_knn(random_query, k, beam_width)
            avg_dist = np.mean(distances)
            print(f"  L={beam_width:3d}: avg distance = {avg_dist:.6f}")


if __name__ == "__main__":
    main()
