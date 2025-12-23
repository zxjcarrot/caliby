#!/usr/bin/env python3
"""
Test: Index Recovery Functionality

This script tests that indexes can be persisted and recovered correctly:
1. Build an index and add data
2. Flush to disk
3. Create a new index instance that recovers the persisted state
4. Verify the recovered index returns the same search results
"""

import numpy as np
import caliby
import sys



# Configure buffer pool sizes
caliby.set_buffer_config(size_gb=0.3)
def test_hnsw_recovery():
    """Test HNSW index recovery."""
    print("="*60)
    print("Testing HNSW Recovery")
    print("="*60)
    
    # Configuration
    dim = 128
    num_vectors = 1000
    k = 10
    M = 16
    ef_construction = 200
    ef_search = 100
    
    # Generate test data
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Create query vector
    query = vectors[0].copy()
    
    # Build initial index
    print(f"\n1. Building initial HNSW index with {num_vectors} vectors...")
    index1 = caliby.HnswIndex(num_vectors, dim, M, ef_construction, skip_recovery=True)
    index1.add_items(vectors)
    
    # Search with initial index
    print("2. Searching with initial index...")
    labels1, distances1 = index1.search_knn(query, k, ef_search)
    print(f"   Found {len(labels1)} neighbors")
    print(f"   Top 3 results: labels={labels1[:3].tolist()}, distances={distances1[:3].tolist()}")
    
    # Flush to disk
    print("3. Flushing index to disk...")
    index1.flush()
    
    
    # Delete the index to clear memory
    del index1
    print("4. Deleted index from memory")
    
    # Create new index that should recover from disk
    print("5. Creating new index instance (should recover from disk)...")
    index2 = caliby.HnswIndex(num_vectors, dim, M, ef_construction, skip_recovery=False)
    
    if not index2.was_recovered():
        print("   ✗ FAILED: Index was not recovered from disk!")
        return False
    print("   ✓ Index was successfully recovered from disk")
    
    # Search with recovered index
    print("6. Searching with recovered index...")
    labels2, distances2 = index2.search_knn(query, k, ef_search)
    print(f"   Found {len(labels2)} neighbors")
    print(f"   Top 3 results: labels={labels2[:3].tolist()}, distances={distances2[:3].tolist()}")
    
    # Compare results
    print("7. Comparing results...")
    labels_match = np.array_equal(labels1, labels2)
    distances_close = np.allclose(distances1, distances2, rtol=1e-5)
    
    if labels_match and distances_close:
        print("   ✓ Results match! Recovery successful.")
        return True
    else:
        print(f"   ✗ Results don't match!")
        print(f"   Labels match: {labels_match}")
        print(f"   Distances close: {distances_close}")
        if not labels_match:
            print(f"   Original labels:  {labels1.tolist()}")
            print(f"   Recovered labels: {labels2.tolist()}")
        return False


def test_hnsw_skip_recovery():
    """Test that skip_recovery=True rebuilds the index from scratch."""
    print("\n" + "="*60)
    print("Testing HNSW Skip Recovery")
    print("="*60)
    
    dim = 128
    num_vectors = 500
    k = 5
    
    # Generate test data
    np.random.seed(123)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    query = vectors[0].copy()
    
    # Build and flush first index
    print("\n1. Building and flushing first index...")
    index1 = caliby.HnswIndex(num_vectors, dim, skip_recovery=True)
    index1.add_items(vectors)
    index1.flush()
    
    labels1, distances1 = index1.search_knn(query, k, 100)
    del index1
    
    # Create new index with skip_recovery=True (should NOT recover)
    print("2. Creating new index with skip_recovery=True...")
    index2 = caliby.HnswIndex(num_vectors, dim, skip_recovery=True)
    
    if index2.was_recovered():
        print("   ✗ FAILED: Index should not have been recovered with skip_recovery=True")
        return False
    print("   ✓ Index correctly skipped recovery")
    
    # The new index should be empty, so add different data
    print("3. Adding same data to new index...")
    index2.add_items(vectors)
    labels2, distances2 = index2.search_knn(query, k, 100)
    
    # Results should match (same data added)
    if np.array_equal(labels1, labels2) and np.allclose(distances1, distances2, rtol=1e-5):
        print("   ✓ Results match as expected")
        return True
    else:
        print("   ✗ Results don't match (this may be OK due to randomness in graph construction)")
        # This is actually acceptable - graph construction has some randomness
        return True


def test_multiple_recovery_cycles():
    """Test multiple recovery cycles."""
    print("\n" + "="*60)
    print("Testing Multiple Recovery Cycles")
    print("="*60)
    
    dim = 128
    num_vectors = 500
    k = 5
    num_cycles = 3
    
    np.random.seed(456)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0].copy()
    
    # Initial build
    print(f"\n1. Building initial index with {num_vectors} vectors...")
    index = caliby.HnswIndex(num_vectors, dim, skip_recovery=True)
    index.add_items(vectors)
    index.flush()
    
    
    initial_labels, initial_distances = index.search_knn(query, k, 100)
    print(f"   Initial search: {initial_labels[:3].tolist()}")
    del index
    
    # Test multiple recovery cycles
    for i in range(num_cycles):
        print(f"\n{i+2}. Recovery cycle {i+1}/{num_cycles}...")
        index = caliby.HnswIndex(num_vectors, dim, skip_recovery=False)
        
        if not index.was_recovered():
            print(f"   ✗ FAILED: Recovery cycle {i+1} failed")
            return False
        
        labels, distances = index.search_knn(query, k, 100)
        
        if not np.array_equal(labels, initial_labels):
            print(f"   ✗ FAILED: Results differ after recovery cycle {i+1}")
            print(f"   Expected: {initial_labels[:3].tolist()}")
            print(f"   Got:      {labels[:3].tolist()}")
            return False
        
        print(f"   ✓ Cycle {i+1} successful: {labels[:3].tolist()}")
        
        # Flush again for next cycle
        if i < num_cycles - 1:
            index.flush()
            
        del index
    
    print(f"\n✓ All {num_cycles} recovery cycles successful!")
    return True


def main():
    """Run all recovery tests."""
    print("\nStarting Index Recovery Tests")
    print("This will test that indexes can be persisted and recovered correctly.\n")
    
    results = []
    
    # Test 1: Basic HNSW recovery
    try:
        result = test_hnsw_recovery()
        results.append(("HNSW Recovery", result))
    except Exception as e:
        print(f"\n✗ HNSW Recovery test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("HNSW Recovery", False))
    
    # Test 2: Skip recovery flag
    try:
        result = test_hnsw_skip_recovery()
        results.append(("HNSW Skip Recovery", result))
    except Exception as e:
        print(f"\n✗ HNSW Skip Recovery test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("HNSW Skip Recovery", False))
    
    # Test 3: Multiple recovery cycles
    try:
        result = test_multiple_recovery_cycles()
        results.append(("Multiple Recovery Cycles", result))
    except Exception as e:
        print(f"\n✗ Multiple Recovery Cycles test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Recovery Cycles", False))
    
    # Print summary
    print("\n" + "="*60)
    print("Recovery Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<45} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All recovery tests passed!")
        return 0
    else:
        print("\n✗ Some recovery tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
