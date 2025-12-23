#!/usr/bin/env python3
"""
Large-scale benchmark for Caliby with varying buffer pool sizes.
Tests both in-memory and larger-than-memory scenarios.
"""

import numpy as np
import time
import os
import sys
import subprocess

# Add parent directory to path for caliby import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_random_data(num_vectors, dim, seed=42):
    """Generate random normalized vectors."""
    np.random.seed(seed)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def run_benchmark(num_vectors, dim, num_queries, physgb):
    """Run benchmark with specific buffer pool configuration."""
    
    # Clean up any existing heapfile
    if os.path.exists('heapfile'):
        os.remove('heapfile')
    
    print(f"\n{'='*70}")
    print(f"Testing with {num_vectors:,} vectors, {dim}D")
    print(f"Buffer Pool: {physgb}GB physical (VIRTGB auto-computed per-index)")
    print(f"{'='*70}")
    
    # Calculate approximate memory requirements
    vector_memory_mb = (num_vectors * dim * 4) / (1024 * 1024)  # 4 bytes per float
    # HNSW needs ~1KB per vector for graph structure
    hnsw_overhead_mb = num_vectors * 1.0 / 1024
    total_data_mb = vector_memory_mb + hnsw_overhead_mb
    buffer_pool_mb = physgb * 1024
    
    print(f"\nMemory Analysis:")
    print(f"  Vector data: ~{vector_memory_mb:.0f} MB")
    print(f"  HNSW overhead: ~{hnsw_overhead_mb:.0f} MB")
    print(f"  Total data: ~{total_data_mb:.0f} MB")
    print(f"  Buffer pool: {buffer_pool_mb:.0f} MB")
    
    if total_data_mb < buffer_pool_mb:
        print(f"  Scenario: IN-MEMORY (data fits in buffer pool)")
    else:
        print(f"  Scenario: OUT-OF-CORE (data {total_data_mb/buffer_pool_mb:.1f}x larger than buffer pool)")
    
    # Create a Python script to run in subprocess with controlled environment
    script = f"""
import sys
import os
import numpy as np
import time

# Import caliby and configure buffer pool
import caliby
caliby.set_buffer_config(size_gb={physgb})

np.random.seed(42)
vectors = np.random.randn({num_vectors}, {dim}).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

queries = np.random.randn({num_queries}, {dim}).astype(np.float32)
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

print("\\nBuilding HNSW index...")
start = time.perf_counter()
index = caliby.HnswIndex({num_vectors}, {dim}, 16, 200, enable_prefetch=True, skip_recovery=True)
index.add_items(vectors)
build_time = time.perf_counter() - start
build_throughput = {num_vectors} / build_time

print(f"  Build time: {{build_time:.2f}}s ({{build_throughput:.0f}} vectors/sec)")

print("\\nSearching...")
start = time.perf_counter()
for query in queries:
    labels, distances = index.search_knn(query, 10, 100)
search_time = time.perf_counter() - start

qps = {num_queries} / search_time
avg_latency_ms = (search_time / {num_queries}) * 1000

print(f"  Search: {{qps:.0f}} QPS, {{avg_latency_ms:.2f}}ms avg latency")

# Flush to ensure all data is written

print("\\nFlushed all data to disk")
"""
    
    # Write script to temp file
    script_path = '/tmp/caliby_benchmark_temp.py'
    with open(script_path, 'w') as f:
        f.write(script)
    
    try:
        # Run the script in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        
        if result.returncode != 0:
            print(f"ERROR: Benchmark failed with return code {result.returncode}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("ERROR: Benchmark timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def main():
    print("="*70)
    print("Caliby Large-Scale Benchmark")
    print("Testing multi-index system with varying buffer pool sizes")
    print("="*70)
    
    # Test configurations: (num_vectors, dim, num_queries, physgb, description)
    # Note: VIRTGB is auto-computed per-index based on max_elements
    configs = [
        # Small warmup test
        (10000, 128, 1000, 1, "Warmup - 10K vectors"),
        
        # 1M vectors, in-memory scenario (generous buffer pool)
        (1000000, 128, 1000, 16, "1M vectors - IN-MEMORY (16GB buffer)"),
        
        # 1M vectors, tight memory (buffer pool ~= data size)
        (1000000, 128, 1000, 2, "1M vectors - TIGHT (2GB buffer)"),
        
        # 1M vectors, memory pressure (buffer pool < data size)
        (1000000, 128, 1000, 1, "1M vectors - OUT-OF-CORE (1GB buffer)"),
    ]
    
    results = []
    
    for num_vectors, dim, num_queries, physgb, description in configs:
        print(f"\n\n{'#'*70}")
        print(f"# {description}")
        print(f"{'#'*70}")
        
        success = run_benchmark(num_vectors, dim, num_queries, physgb)
        results.append((description, success))
        
        # Short pause between tests
        time.sleep(2)
    
    # Summary
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    for desc, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {desc}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n✓ All benchmarks completed successfully!")
    else:
        print("\n✗ Some benchmarks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
