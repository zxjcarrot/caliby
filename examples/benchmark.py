#!/usr/bin/env python3
"""
Example: Benchmarking Caliby indexes

This example demonstrates:
1. Building indexes with different parameters
2. Measuring throughput and latency
3. Comparing HNSW vs DiskANN performance
"""

import numpy as np
import time
import tempfile
import os
import subprocess
import sys
import urllib.request
import tarfile
import struct

try:
    import caliby
    # Configure buffer pool sizes
    caliby.set_buffer_config(size_gb=3)
except ImportError:
    print("Error: caliby not installed. Build and install with:")
    print("  pip install -e .")
    exit(1)


def generate_random_data(num_vectors, dim, seed=42):
    """Generate random normalized vectors."""
    np.random.seed(seed)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def read_fvecs(filename):
    """Read .fvecs file format (used by SIFT dataset)."""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            # Read vector
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def read_ivecs(filename):
    """Read .ivecs file format (used for ground truth)."""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            # Read vector
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
        return np.array(vectors, dtype=np.int32)


def download_sift1m(data_dir='./sift1m'):
    """Download and extract SIFT1M dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    
    base_file = os.path.join(data_dir, 'sift_base.fvecs')
    query_file = os.path.join(data_dir, 'sift_query.fvecs')
    groundtruth_file = os.path.join(data_dir, 'sift_groundtruth.ivecs')
    
    # Check if already downloaded
    if os.path.exists(base_file) and os.path.exists(query_file) and os.path.exists(groundtruth_file):
        print(f"SIFT1M dataset already exists in {data_dir}")
        return data_dir
    
    print("Downloading SIFT1M dataset...")
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    tar_path = os.path.join(data_dir, 'sift.tar.gz')
    
    try:
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        
        print("  Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # Move files to data_dir root
        sift_dir = os.path.join(data_dir, 'sift')
        if os.path.exists(sift_dir):
            for filename in os.listdir(sift_dir):
                src = os.path.join(sift_dir, filename)
                dst = os.path.join(data_dir, filename)
                if not os.path.exists(dst):
                    os.rename(src, dst)
            os.rmdir(sift_dir)
        
        # Clean up tar file
        os.remove(tar_path)
        
        print(f"  SIFT1M dataset downloaded to {data_dir}")
        return data_dir
    except Exception as e:
        print(f"  Warning: Failed to download SIFT1M dataset: {e}")
        print("  Falling back to random data generation")
        return None


def load_sift1m_data(data_dir='./sift1m', num_vectors=None, num_queries=None):
    """Load SIFT1M dataset or generate random data as fallback."""
    base_file = os.path.join(data_dir, 'sift_base.fvecs')
    query_file = os.path.join(data_dir, 'sift_query.fvecs')
    groundtruth_file = os.path.join(data_dir, 'sift_groundtruth.ivecs')
    
    if os.path.exists(base_file) and os.path.exists(query_file):
        print(f"Loading SIFT1M dataset from {data_dir}...")
        base_vectors = read_fvecs(base_file)
        query_vectors = read_fvecs(query_file)
        
        # Load ground truth if available
        groundtruth = None
        if os.path.exists(groundtruth_file):
            groundtruth = read_ivecs(groundtruth_file)
            if num_queries and num_queries < len(groundtruth):
                groundtruth = groundtruth[:num_queries]
        
        if num_vectors and num_vectors < len(base_vectors):
            base_vectors = base_vectors[:num_vectors]
        
        if num_queries and num_queries < len(query_vectors):
            query_vectors = query_vectors[:num_queries]
        
        print(f"  Loaded {len(base_vectors)} base vectors, {len(query_vectors)} query vectors")
        if groundtruth is not None:
            print(f"  Ground truth: {len(groundtruth)} queries")
        print(f"  Dimension: {base_vectors.shape[1]}")
        return base_vectors, query_vectors, groundtruth
    else:
        print("SIFT1M dataset not found, generating random data...")
        dim = 128
        num_vecs = num_vectors or 10000
        base_vectors = generate_random_data(num_vecs, dim)
        query_vectors = generate_random_data(num_queries, dim, seed=123)
        return base_vectors, query_vectors, None


def benchmark_hnsw(vectors, queries, k=10, M=16, ef_construction=100, ef_search=50, profile=False):
    """Benchmark HNSW index."""
    dim = vectors.shape[1]
    num_vectors = vectors.shape[0]
    
    # Build index
    print(f"  Building HNSW index (M={M}, ef_construction={ef_construction})...")
    index = caliby.HnswIndex(num_vectors, dim, M, ef_construction, enable_prefetch=True, skip_recovery=True)
    
    build_start = time.perf_counter()
    index.add_points(vectors)
    build_time = time.perf_counter() - build_start
    
    build_throughput = num_vectors / build_time
    print(f"    Build time: {build_time:.2f}s ({build_throughput:.0f} vectors/sec)")
    
    # Search
    print(f"  Searching (ef={ef_search}, k={k})...")
    
    if profile:
        print("  Starting perf profiling for HNSW search...")
        pid = os.getpid()
        perf_proc = subprocess.Popen(
            ['perf', 'record', '-g', '-o', 'perf_hnsw_search.data', '-p', str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    search_start = time.perf_counter()
    labels, distances = index.search_knn_parallel(queries, k, ef_search, num_threads=1)
    search_time = time.perf_counter() - search_start
    results = list(zip(distances, labels))
    
    if profile:
        perf_proc.terminate()
        perf_proc.wait()
        print("  Perf data saved to: perf_hnsw_search.data")
    
    qps = len(queries) / search_time
    avg_latency_us = (search_time / len(queries)) * 1e6
    print(f"    QPS: {qps:.0f}, Avg latency: {avg_latency_us:.0f}µs")
    
    return {
        'build_time': build_time,
        'build_throughput': build_throughput,
        'qps': qps,
        'avg_latency_us': avg_latency_us,
        'results': results
    }


def benchmark_diskann(vectors, queries, k=10, R=64, alpha=1.2, L=50, profile=False):
    """Benchmark DiskANN index."""
    dim = vectors.shape[1]
    num_vectors = vectors.shape[0]
    
    # Build index
    print(f"  Building DiskANN index (R={R}, alpha={alpha})...")
    index = caliby.DiskANN(dim, num_vectors, R, is_dynamic=False)
    
    # Create build parameters
    build_params = caliby.BuildParams()
    build_params.L_build = L
    build_params.alpha = alpha
    build_params.num_threads = 32
    
    # Prepare tags (one empty tag per vector)
    tags = [[] for _ in range(num_vectors)]
    
    build_start = time.perf_counter()
    index.build(vectors, tags, build_params)
    build_time = time.perf_counter() - build_start
    
    build_throughput = num_vectors / build_time
    print(f"    Build time: {build_time:.2f}s ({build_throughput:.0f} vectors/sec)")
    
    # Search
    search_params = caliby.SearchParams(L)
    search_params.beam_width = 2
    
    print(f"  Searching (L={L}, k={k})...")
    
    if profile:
        print("  Starting perf profiling for DiskANN search...")
        pid = os.getpid()
        perf_proc = subprocess.Popen(
            ['perf', 'record', '-g', '-o', 'perf_diskann_search.data', '-p', str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    search_start = time.perf_counter()
    labels, distances = index.search_knn_parallel(queries, k, search_params, num_threads=1)
    search_time = time.perf_counter() - search_start
    results = list(zip(distances, labels))
    
    if profile:
        perf_proc.terminate()
        perf_proc.wait()
        print("  Perf data saved to: perf_diskann_search.data")
    
    qps = len(queries) / search_time
    avg_latency_us = (search_time / len(queries)) * 1e6
    print(f"    QPS: {qps:.0f}, Avg latency: {avg_latency_us:.0f}µs")
    
    return {
        'build_time': build_time,
        'build_throughput': build_throughput,
        'qps': qps,
        'avg_latency_us': avg_latency_us,
        'results': results
    }


def compute_recall(results1, results2, k=10):
    """Compute recall between two result sets."""
    total_recall = 0
    for (_, labels1), (_, labels2) in zip(results1, results2):
        set1 = set(labels1[:k])
        set2 = set(labels2[:k])
        recall = len(set1 & set2) / k
        total_recall += recall
    return total_recall / len(results1)


def compute_recall_at_k(results, groundtruth, k=10):
    """Compute recall@k against ground truth."""
    if groundtruth is None:
        return None
    
    total_recall = 0
    for i, (_, labels) in enumerate(results):
        predicted = set(labels[:k])
        actual = set(groundtruth[i][:k])
        recall = len(predicted & actual) / k
        total_recall += recall
    return total_recall / len(results)


def main():
    # Configuration
    k = 10
    enable_profiling = '--profile' in sys.argv
    
    print("="*60)
    print("Caliby Benchmark")
    print("="*60)
    
    # Try to download SIFT1M dataset if not present
    data_dir = download_sift1m()
    
    # Load data (SIFT1M or fallback to random)
    vectors, queries, groundtruth = load_sift1m_data('./sift1m')
    dim = vectors.shape[1]
    
    print(f"Dataset: {len(vectors)} vectors, {dim} dimensions")
    print(f"Queries: {len(queries)}, k={k}")
    if groundtruth is not None:
        print(f"Ground truth available: Yes")
    else:
        print(f"Ground truth available: No (using cross-validation)")
    if enable_profiling:
        print("Profiling: ENABLED")
    print()
    
    # Benchmark HNSW
    print("\n" + "-"*40)
    print("HNSW Benchmark")
    print("-"*40)
    hnsw_results = benchmark_hnsw(vectors, queries, k=k, profile=enable_profiling)
    
    # Benchmark DiskANN
    print("\n" + "-"*40)
    print("DiskANN Benchmark")
    print("-"*40)
    diskann_results = benchmark_diskann(vectors, queries, k=k, profile=enable_profiling)
    
    # Compute recall
    if groundtruth is not None:
        hnsw_recall = compute_recall_at_k(hnsw_results['results'], groundtruth, k)
        diskann_recall = compute_recall_at_k(diskann_results['results'], groundtruth, k)
    else:
        hnsw_recall = None
        diskann_recall = None
    
    # Compare results
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Metric':<25} {'HNSW':>15} {'DiskANN':>15}")
    print("-"*55)
    print(f"{'Build Throughput (vec/s)':<25} {hnsw_results['build_throughput']:>15,.0f} {diskann_results['build_throughput']:>15,.0f}")
    print(f"{'Search QPS':<25} {hnsw_results['qps']:>15,.0f} {diskann_results['qps']:>15,.0f}")
    print(f"{'Avg Latency (µs)':<25} {hnsw_results['avg_latency_us']:>15,.0f} {diskann_results['avg_latency_us']:>15,.0f}")
    
    if hnsw_recall is not None and diskann_recall is not None:
        print(f"{'Recall@' + str(k):<25} {hnsw_recall:>15.2%} {diskann_recall:>15.2%}")
    
    # Compute recall agreement between the two indexes
    recall_agreement = compute_recall(hnsw_results['results'], diskann_results['results'], k)
    print(f"\nRecall agreement between HNSW and DiskANN: {recall_agreement:.2%}")
    
    if enable_profiling:
        print("\n" + "="*60)
        print("Profiling data generated:")
        print("  - perf_hnsw_search.data")
        print("  - perf_diskann_search.data")
        print("\nView with:")
        print("  perf report -i perf_hnsw_search.data")
        print("  perf report -i perf_diskann_search.data")


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python3 benchmark.py [--profile]")
        print("  --profile: Enable perf profiling for search phases")
        sys.exit(0)
    main()
