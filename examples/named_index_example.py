#!/usr/bin/env python3
"""
Example: Using Named Indexes in Caliby

This example demonstrates how to use the index naming feature to create
and manage multiple HNSW indexes with unique, human-readable names.
"""

import numpy as np
import sys
import os

# Add build directory to path
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

import caliby


# Configure buffer pool sizes
caliby.set_buffer_config(size_gb=0.3)
def main():
    print("=" * 60)
    print("Named Index Example for Caliby")
    print("=" * 60)
    
    # Create multiple indexes with descriptive names
    indexes = {}
    
    # Example 1: User embeddings
    print("\n1. Creating 'user_embeddings' index...")
    user_index = caliby.HnswIndex(
        max_elements=1000,
        dim=128,
        M=16,
        ef_construction=200,
        skip_recovery=True,
        index_id=1,
        name="user_embeddings"
    )
    print(f"   Created index with name: '{user_index.get_name()}'")
    
    # Add some vectors
    user_vectors = np.random.randn(500, 128).astype(np.float32)
    user_index.add_points(user_vectors)
    print(f"   Added {len(user_vectors)} user vectors")
    
    indexes["user_embeddings"] = user_index
    
    # Example 2: Product embeddings
    print("\n2. Creating 'product_embeddings' index...")
    product_index = caliby.HnswIndex(
        max_elements=2000,
        dim=256,
        M=16,
        ef_construction=200,
        skip_recovery=True,
        index_id=2,
        name="product_embeddings"
    )
    print(f"   Created index with name: '{product_index.get_name()}'")
    
    product_vectors = np.random.randn(800, 256).astype(np.float32)
    product_index.add_points(product_vectors)
    print(f"   Added {len(product_vectors)} product vectors")
    
    indexes["product_embeddings"] = product_index
    
    # Example 3: Document embeddings
    print("\n3. Creating 'document_embeddings' index...")
    doc_index = caliby.HnswIndex(
        max_elements=5000,
        dim=512,
        M=8,
        ef_construction=150,
        skip_recovery=True,
        index_id=3,
        name="document_embeddings"
    )
    print(f"   Created index with name: '{doc_index.get_name()}'")
    
    doc_vectors = np.random.randn(1200, 512).astype(np.float32)
    doc_index.add_points(doc_vectors)
    print(f"   Added {len(doc_vectors)} document vectors")
    
    indexes["document_embeddings"] = doc_index
    
    # Demonstrate querying by name
    print("\n" + "=" * 60)
    print("Querying indexes by name:")
    print("=" * 60)
    
    for name, index in indexes.items():
        print(f"\n{name}:")
        print(f"  - Index name: {index.get_name()}")
        print(f"  - Dimension: {index.get_dim()}")
        
        # Perform a search
        if name == "user_embeddings":
            query = user_vectors[0]
        elif name == "product_embeddings":
            query = product_vectors[0]
        else:
            query = doc_vectors[0]
        
        labels, distances = index.search_knn(query, k=5, ef_search=50)
        print(f"  - Search results (k=5): {labels[:5]}")
        print(f"  - Top distance: {distances[0]:.6f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Demonstrate practical use case: Multi-tenant vector database
    print("\n" + "=" * 60)
    print("Multi-Tenant Example:")
    print("=" * 60)
    
    tenants = ["acme_corp", "globex_inc", "initech_llc"]
    tenant_indexes = {}
    
    for i, tenant in enumerate(tenants):
        print(f"\nCreating index for {tenant}...")
        tenant_index = caliby.HnswIndex(
            max_elements=500,
            dim=128,
            M=16,
            ef_construction=100,
            skip_recovery=True,
            index_id=10 + i,
            name=f"tenant_{tenant}"
        )
        
        # Add tenant-specific vectors
        vectors = np.random.randn(200, 128).astype(np.float32)
        tenant_index.add_points(vectors)
        
        tenant_indexes[tenant] = {
            "index": tenant_index,
            "vectors": vectors
        }
        
        print(f"  ✓ {tenant_index.get_name()} ready with {len(vectors)} vectors")
    
    # Simulate tenant-specific queries
    print("\nPerforming tenant-specific searches:")
    for tenant, data in tenant_indexes.items():
        index = data["index"]
        vectors = data["vectors"]
        
        query = vectors[0]
        labels, distances = index.search_knn(query, k=3, ef_search=50)
        
        print(f"  {index.get_name()}: Found {len(labels)} results")

if __name__ == "__main__":
    main()
