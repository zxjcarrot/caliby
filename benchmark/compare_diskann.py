#!/usr/bin/env python3
"""
DiskANN Benchmark: Compare Caliby DiskANN vs Microsoft DiskANN (diskannpy)

This benchmark compares two DiskANN implementations:
- Caliby: Buffer-managed DiskANN with disk persistence
- Microsoft DiskANN (diskannpy): Official Microsoft implementation with StaticDiskIndex

Uses the DEEP10M dataset (10M vectors, 96 dimensions) for evaluation.

Metrics measured:
- Index build time
- Index size (disk)
- Search throughput (QPS)
- Search latency (P50, P95, P99)
- Recall@10 accuracy
"""

import numpy as np
import time
import os
import sys
import struct
import tempfile
import urllib.request
import tarfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import argparse

# Add parent directory to path to import local caliby build
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing all libraries
LIBS_AVAILABLE = {}

try:
    import caliby
    caliby.set_buffer_config(size_gb=4.0)
    LIBS_AVAILABLE['caliby'] = True
except ImportError:
    print("Warning: caliby not available")
    LIBS_AVAILABLE['caliby'] = False

try:
    import diskannpy
    LIBS_AVAILABLE['diskannpy'] = True
except ImportError:
    print("Warning: diskannpy not available (install: pip install diskannpy)")
    LIBS_AVAILABLE['diskannpy'] = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    library: str
    build_time: float
    index_size_mb: float
    qps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    recall_at_10: float
    params: Dict = field(default_factory=dict)


def read_fbin(filename):
    """Read .fbin file format (used by DEEP dataset)."""
    with open(filename, 'rb') as f:
        # Read header: num_vectors (4 bytes), dimension (4 bytes)
        num_vectors = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # Read all vectors at once
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        return data.reshape(num_vectors, dim)


def read_ibin(filename):
    """Read .ibin file format (used for ground truth in DEEP dataset)."""
    with open(filename, 'rb') as f:
        # Read header: num_vectors (4 bytes), dimension (4 bytes)
        num_vectors = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # Read all vectors at once
        data = np.fromfile(f, dtype=np.int32, count=num_vectors * dim)
        return data.reshape(num_vectors, dim)


def read_fvecs(filename):
    """Read .fvecs file format (used by SIFT dataset)."""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def read_ivecs(filename):
    """Read .ivecs file format (used for ground truth)."""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
        return np.array(vectors, dtype=np.int32)


def detect_file_format(filename):
    """Detect if file is .fvecs/.ivecs or .fbin/.ibin format."""
    if filename.endswith('.fvecs') or filename.endswith('.ivecs'):
        return 'vecs'
    elif filename.endswith('.fbin') or filename.endswith('.ibin'):
        return 'bin'
    else:
        raise ValueError(f"Unknown file format: {filename}")


def read_base_vectors(filename):
    """Auto-detect format and read base vectors."""
    fmt = detect_file_format(filename)
    if fmt == 'vecs':
        return read_fvecs(filename)
    elif fmt == 'bin':
        return read_fbin(filename)


def read_query_vectors(filename):
    """Auto-detect format and read query vectors."""
    return read_base_vectors(filename)  # Same logic


def read_groundtruth(filename):
    """Auto-detect format and read ground truth."""
    fmt = detect_file_format(filename)
    if fmt == 'vecs':
        return read_ivecs(filename)
    elif fmt == 'bin':
        return read_ibin(filename)


def download_deep10m(data_dir='./deep10M'):
    """Download DEEP10M dataset if not present.
    
    Dataset info:
    - Base vectors: deep10M_base.fbin (10M vectors, 96 dim)
    - Query vectors: deep10M_query.fbin (10K queries)
    - Ground truth: deep10M_groundtruth.ibin (100-NN for each query)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Check for both .fbin and .fvecs formats
    base_file_bin = os.path.join(data_dir, 'deep10M_base.fbin')
    query_file_bin = os.path.join(data_dir, 'deep10M_query.fbin')
    groundtruth_file_bin = os.path.join(data_dir, 'deep10M_groundtruth.ibin')
    
    # Check if files exist
    if os.path.exists(base_file_bin) and os.path.exists(query_file_bin) and os.path.exists(groundtruth_file_bin):
        print(f"✓ DEEP10M dataset already exists in {data_dir}")
        return data_dir, '.fbin'
    
    print("DEEP10M dataset not found.")
    print("Please download from: https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search")
    print(f"\nExpected files in {data_dir}:")
    print("  - deep10M_base.fbin (10M base vectors, 96 dim)")
    print("  - deep10M_query.fbin (10K query vectors)")
    print("  - deep10M_groundtruth.ibin (ground truth)")
    print("\nAlternative: Use a smaller subset for testing:")
    print("  python compare_diskann.py --data-dir ./sift1m --use-sift")
    sys.exit(1)


def load_deep10m_data(data_dir='./deep10M', max_vectors=None, use_sift=False):
    """Load dataset (DEEP10M or SIFT1M)."""
    if use_sift:
        print("\nLoading SIFT1M dataset...")
        base_file = os.path.join(data_dir, 'sift_base.fvecs')
        query_file = os.path.join(data_dir, 'sift_query.fvecs')
        groundtruth_file = os.path.join(data_dir, 'sift_groundtruth.ivecs')
        
        if not os.path.exists(base_file):
            print(f"Error: SIFT1M not found. Expected: {base_file}")
            print("Run: python compare_hnsw.py to download SIFT1M first")
            sys.exit(1)
            
        base_vectors = read_fvecs(base_file)
        query_vectors = read_fvecs(query_file)
        groundtruth = read_ivecs(groundtruth_file)
    else:
        print("\nLoading DEEP10M dataset...")
        # Try multiple file naming conventions
        base_candidates = [
            os.path.join(data_dir, 'deep10M_base.fbin'),
            os.path.join(data_dir, 'base.10M.fbin'),
        ]
        query_candidates = [
            os.path.join(data_dir, 'deep10M_query.fbin'),
            os.path.join(data_dir, 'query.public.10K.fbin'),
        ]
        gt_candidates = [
            os.path.join(data_dir, 'deep10M_groundtruth.ibin'),
            os.path.join(data_dir, 'groundtruth.ibin'),
        ]
        
        base_file = next((f for f in base_candidates if os.path.exists(f)), None)
        query_file = next((f for f in query_candidates if os.path.exists(f)), None)
        groundtruth_file = next((f for f in gt_candidates if os.path.exists(f)), None)
        
        if not base_file or not query_file or not groundtruth_file:
            download_deep10m(data_dir)
            # Try again after download message
            sys.exit(1)
        
        base_vectors = read_fbin(base_file)
        query_vectors = read_fbin(query_file)
        groundtruth = read_ibin(groundtruth_file)
    
    # Optionally limit number of vectors for testing
    if max_vectors is not None and max_vectors < len(base_vectors):
        print(f"  Limiting to {max_vectors} vectors for testing...")
        base_vectors = base_vectors[:max_vectors]
        # Recompute ground truth for limited vectors
        print("  Note: Ground truth may be inaccurate for limited vectors")
    
    print(f"  Base vectors: {base_vectors.shape}")
    print(f"  Query vectors: {query_vectors.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    return base_vectors, query_vectors, groundtruth


def compute_recall_at_k(results: np.ndarray, groundtruth: np.ndarray, k: int = 10) -> float:
    """
    Compute Recall@k for search results.
    
    Args:
        results: Search results of shape (n_queries, k) containing indices
        groundtruth: Ground truth of shape (n_queries, gt_k) containing indices
        k: Number of top results to consider
    
    Returns:
        Average recall@k across all queries
    """
    n_queries = results.shape[0]
    recalls = []
    
    for i in range(n_queries):
        result_set = set(results[i, :k])
        gt_set = set(groundtruth[i, :k])
        recall = len(result_set & gt_set) / k
        recalls.append(recall)
    
    return np.mean(recalls)


def get_directory_size(path):
    """Calculate total size of directory in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total / (1024 * 1024)  # Convert to MB


def benchmark_caliby_diskann(base_vectors, query_vectors, groundtruth, params, data_dir):
    """Benchmark Caliby DiskANN implementation."""
    print("\n" + "="*70)
    print("Benchmarking Caliby DiskANN")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    # Create temporary directory for index
    index_dir = os.path.join(data_dir, 'caliby_diskann_index')
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  R={params['R']}, L_build={params['L_build']}, alpha={params['alpha']}")
    
    # Open caliby
    caliby.open(index_dir, cleanup_if_exist=True)
    
    start_time = time.time()
    
    # Create DiskANN index
    index = caliby.DiskANN(
        dimensions=dim,
        max_elements=num_vectors + 1000,
        R_max_degree=params['R'],
        is_dynamic=False
    )
    
    # Create tags (each vector gets a single tag based on its index mod 100)
    tags = [[i % 100] for i in range(num_vectors)]
    
    # Build with params
    build_params = caliby.BuildParams()
    build_params.L_build = params['L_build']
    build_params.alpha = params['alpha']
    build_params.num_threads = params.get('num_threads', 0)
    
    print(f"  Building Vamana graph...")
    index.build(base_vectors, tags, build_params)
    
    build_time = time.time() - start_time
    print(f"✓ Build time: {build_time:.2f}s")
    
    # Flush to get accurate size
    caliby.flush_storage()
    
    # Get index size
    index_size_mb = get_directory_size(index_dir)
    print(f"✓ Index size: {index_size_mb:.2f} MB")
    
    # Create search params
    search_params = caliby.SearchParams(L_search=params['L_search'])
    search_params.beam_width = params.get('beam_width', 2)
    
    # Warm-up
    print("\nWarming up...")
    for i in range(min(100, num_queries)):
        _ = index.search(query_vectors[i], K=10, params=search_params)
    
    # Benchmark search
    print(f"Running search benchmark (L_search={params['L_search']})...")
    latencies = []
    all_results = []
    
    for i in range(num_queries):
        start = time.perf_counter()
        labels, distances = index.search(query_vectors[i], K=10, params=search_params)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(labels.astype(np.int32))
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries", end='\r')
    
    print(f"  Processed {num_queries}/{num_queries} queries")
    
    # Calculate metrics
    latencies = np.array(latencies)
    results_array = np.array(all_results)
    qps = num_queries / (np.sum(latencies) / 1000)
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    
    # Cleanup
    caliby.close()
    
    return BenchmarkResult(
        library='Caliby DiskANN',
        build_time=build_time,
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        recall_at_10=recall,
        params=params
    )


def benchmark_diskannpy(base_vectors, query_vectors, groundtruth, params, data_dir):
    """Benchmark Microsoft DiskANN (diskannpy) using StaticDiskIndex."""
    print("\n" + "="*70)
    print("Benchmarking Microsoft DiskANN (diskannpy) - StaticDiskIndex")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    # Create temporary directory for index
    index_dir = os.path.join(data_dir, 'diskannpy_index')
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  R={params['R']}, complexity={params['L_build']}")
    
    start_time = time.time()
    
    # Build disk index using diskannpy
    # Note: build_disk_index requires search_memory_maximum and build_memory_maximum
    diskannpy.build_disk_index(
        data=base_vectors,
        distance_metric='l2',
        index_directory=index_dir,
        complexity=params['L_build'],
        graph_degree=params['R'],
        search_memory_maximum=4.0,  # 4GB for search
        build_memory_maximum=8.0,   # 8GB for build
        num_threads=params.get('num_threads', 0),
        pq_disk_bytes=0,  # Store uncompressed data
        index_prefix='ann'
    )
    
    build_time = time.time() - start_time
    print(f"✓ Build time: {build_time:.2f}s")
    
    # Get index size
    index_size_mb = get_directory_size(index_dir)
    print(f"✓ Index size: {index_size_mb:.2f} MB")
    
    # Load the index for searching
    print("\nLoading StaticDiskIndex...")
    index = diskannpy.StaticDiskIndex(
        index_directory=index_dir,
        num_threads=params.get('num_threads', 0),
        num_nodes_to_cache=100000,  # Cache nodes for better performance
        cache_mechanism=1,
        index_prefix='ann'
    )
    
    # Warm-up
    print("Warming up...")
    for i in range(min(100, num_queries)):
        _ = index.search(
            query=query_vectors[i],
            k_neighbors=10,
            complexity=params['L_search'],
            beam_width=params.get('beam_width', 2)
        )
    
    # Benchmark search
    print(f"Running search benchmark (L_search={params['L_search']})...")
    latencies = []
    all_results = []
    
    for i in range(num_queries):
        start = time.perf_counter()
        result = index.search(
            query=query_vectors[i],
            k_neighbors=10,
            complexity=params['L_search'],
            beam_width=params.get('beam_width', 2)
        )
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(result.identifiers.astype(np.int32))
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries", end='\r')
    
    print(f"  Processed {num_queries}/{num_queries} queries")
    
    # Calculate metrics
    latencies = np.array(latencies)
    results_array = np.array(all_results)
    qps = num_queries / (np.sum(latencies) / 1000)
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    
    return BenchmarkResult(
        library='diskannpy (StaticDisk)',
        build_time=build_time,
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        recall_at_10=recall,
        params=params
    )


def benchmark_diskannpy_batch(base_vectors, query_vectors, groundtruth, params, data_dir):
    """Benchmark Microsoft DiskANN (diskannpy) using batch search."""
    print("\n" + "="*70)
    print("Benchmarking Microsoft DiskANN (diskannpy) - Batch Search")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    # Use existing index if available
    index_dir = os.path.join(data_dir, 'diskannpy_index')
    
    if not os.path.exists(index_dir):
        print("Index not found, building first...")
        diskannpy.build_disk_index(
            data=base_vectors,
            distance_metric='l2',
            index_directory=index_dir,
            complexity=params['L_build'],
            graph_degree=params['R'],
            search_memory_maximum=4.0,
            build_memory_maximum=8.0,
            num_threads=params.get('num_threads', 0),
            pq_disk_bytes=0,
            index_prefix='ann'
        )
    
    # Get index size
    index_size_mb = get_directory_size(index_dir)
    
    # Load the index
    print("Loading StaticDiskIndex for batch search...")
    index = diskannpy.StaticDiskIndex(
        index_directory=index_dir,
        num_threads=params.get('num_threads', 0),
        num_nodes_to_cache=100000,
        cache_mechanism=1,
        index_prefix='ann'
    )
    
    # Warm-up with batch
    print("Warming up...")
    _ = index.batch_search(
        queries=query_vectors[:100],
        k_neighbors=10,
        complexity=params['L_search'],
        num_threads=params.get('num_threads', 0),
        beam_width=params.get('beam_width', 2)
    )
    
    # Benchmark batch search
    print(f"Running batch search benchmark (L_search={params['L_search']})...")
    
    start = time.perf_counter()
    result = index.batch_search(
        queries=query_vectors,
        k_neighbors=10,
        complexity=params['L_search'],
        num_threads=params.get('num_threads', 0),
        beam_width=params.get('beam_width', 2)
    )
    total_time = time.perf_counter() - start
    
    # Calculate metrics
    results_array = result.identifiers.astype(np.int32)
    qps = num_queries / total_time
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    avg_latency = (total_time / num_queries) * 1000  # ms
    
    print(f"  Batch search completed in {total_time:.2f}s")
    
    return BenchmarkResult(
        library='diskannpy (Batch)',
        build_time=0.0,  # Already built
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=avg_latency,
        latency_p95=avg_latency,
        latency_p99=avg_latency,
        recall_at_10=recall,
        params=params
    )


def benchmark_caliby_batch(base_vectors, query_vectors, groundtruth, params, data_dir):
    """Benchmark Caliby DiskANN using parallel batch search."""
    print("\n" + "="*70)
    print("Benchmarking Caliby DiskANN - Batch Search")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    index_dir = os.path.join(data_dir, 'caliby_diskann_index')
    
    # Check if we need to rebuild
    need_rebuild = not os.path.exists(index_dir)
    
    if need_rebuild:
        print("Index not found, building first...")
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        os.makedirs(index_dir)
        
        caliby.open(index_dir, cleanup_if_exist=True)
        
        index = caliby.DiskANN(
            dimensions=dim,
            max_elements=num_vectors + 1000,
            R_max_degree=params['R'],
            is_dynamic=False
        )
        
        tags = [[i % 100] for i in range(num_vectors)]
        build_params = caliby.BuildParams()
        build_params.L_build = params['L_build']
        build_params.alpha = params['alpha']
        build_params.num_threads = params.get('num_threads', 0)
        
        index.build(base_vectors, tags, build_params)
        caliby.flush_storage()
    else:
        caliby.open(index_dir, cleanup_if_exist=False)
        
        index = caliby.DiskANN(
            dimensions=dim,
            max_elements=num_vectors + 1000,
            R_max_degree=params['R'],
            is_dynamic=False
        )
    
    index_size_mb = get_directory_size(index_dir)
    
    # Create search params
    search_params = caliby.SearchParams(L_search=params['L_search'])
    search_params.beam_width = params.get('beam_width', 2)
    
    # Warm-up with batch
    print("Warming up...")
    _ = index.search_knn_parallel(
        queries=query_vectors[:100],
        K=10,
        params=search_params,
        num_threads=params.get('num_threads', 0)
    )
    
    # Benchmark batch search
    print(f"Running batch search benchmark (L_search={params['L_search']})...")
    
    start = time.perf_counter()
    labels, distances = index.search_knn_parallel(
        queries=query_vectors,
        K=10,
        params=search_params,
        num_threads=params.get('num_threads', 0)
    )
    total_time = time.perf_counter() - start
    
    # Calculate metrics
    results_array = labels.astype(np.int32)
    qps = num_queries / total_time
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    avg_latency = (total_time / num_queries) * 1000  # ms
    
    print(f"  Batch search completed in {total_time:.2f}s")
    
    caliby.close()
    
    return BenchmarkResult(
        library='Caliby (Batch)',
        build_time=0.0,
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=avg_latency,
        latency_p95=avg_latency,
        latency_p99=avg_latency,
        recall_at_10=recall,
        params=params
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print comparison table of benchmark results."""
    print("\n" + "="*110)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*110)
    
    # Header
    print(f"\n{'Library':<25} {'Build(s)':<12} {'Size(MB)':<12} {'QPS':<12} "
          f"{'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Recall@10':<12}")
    print("-" * 110)
    
    # Sort by QPS (descending)
    results_sorted = sorted(results, key=lambda x: x.qps, reverse=True)
    
    for r in results_sorted:
        print(f"{r.library:<25} {r.build_time:<12.2f} {r.index_size_mb:<12.2f} "
              f"{r.qps:<12.1f} {r.latency_p50:<10.3f} {r.latency_p95:<10.3f} "
              f"{r.latency_p99:<10.3f} {r.recall_at_10:<12.4f}")
    
    print("\n" + "="*110)
    
    # Find best in each category
    if len(results) > 1:
        best_qps = max(results, key=lambda x: x.qps)
        best_recall = max(results, key=lambda x: x.recall_at_10)
        best_latency = min(results, key=lambda x: x.latency_p50)
        smallest_index = min(results, key=lambda x: x.index_size_mb)
        
        # Only show build time winner for results with build_time > 0
        build_results = [r for r in results if r.build_time > 0]
        
        print("\nWINNERS:")
        print(f"  🏆 Highest QPS:        {best_qps.library} ({best_qps.qps:.1f} queries/sec)")
        print(f"  🏆 Best Recall@10:     {best_recall.library} ({best_recall.recall_at_10:.4f})")
        print(f"  🏆 Lowest P50 Latency: {best_latency.library} ({best_latency.latency_p50:.3f} ms)")
        print(f"  🏆 Smallest Index:     {smallest_index.library} ({smallest_index.index_size_mb:.2f} MB)")
        if build_results:
            fastest_build = min(build_results, key=lambda x: x.build_time)
            print(f"  🏆 Fastest Build:      {fastest_build.library} ({fastest_build.build_time:.2f} s)")
    
    print("\n" + "="*110)


def main():
    parser = argparse.ArgumentParser(description='DiskANN Benchmark: Caliby vs Microsoft DiskANN')
    parser.add_argument('--data-dir', default='./deep10M', help='Directory for dataset')
    parser.add_argument('--use-sift', action='store_true', help='Use SIFT1M instead of DEEP10M')
    parser.add_argument('--max-vectors', type=int, default=None, help='Limit number of base vectors (for testing)')
    parser.add_argument('--R', type=int, default=64, help='Max graph degree (R)')
    parser.add_argument('--L-build', type=int, default=100, help='Build complexity (L_build)')
    parser.add_argument('--L-search', type=int, default=100, help='Search complexity (L_search)')
    parser.add_argument('--alpha', type=float, default=1.2, help='Alpha parameter for Vamana')
    parser.add_argument('--beam-width', type=int, default=2, help='Beam width for disk search')
    parser.add_argument('--num-threads', type=int, default=0, help='Number of threads (0=auto)')
    parser.add_argument('--skip', type=str,
                        help='Libraries to skip (comma separated, e.g., "caliby,diskannpy")')
    parser.add_argument('--batch-only', action='store_true', help='Only run batch search benchmarks')
    args = parser.parse_args()
    
    print("="*110)
    print("DISKANN BENCHMARK: Caliby vs Microsoft DiskANN (diskannpy)")
    print("="*110)
    
    dataset_name = "SIFT1M" if args.use_sift else "DEEP10M"
    print(f"\nDataset: {dataset_name}")
    print(f"Parameters: R={args.R}, L_build={args.L_build}, L_search={args.L_search}, alpha={args.alpha}")
    
    # Check available libraries
    skip_libs = set()
    if args.skip:
        for lib in args.skip.replace(',', ' ').split():
            lib = lib.strip().lower()
            if lib in ['caliby', 'diskannpy']:
                skip_libs.add(lib)
    
    enabled_libs = {lib: available and lib not in skip_libs 
                    for lib, available in LIBS_AVAILABLE.items()}
    
    print(f"\nLibraries enabled:")
    for lib, enabled in enabled_libs.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {lib:<15} {status}")
    
    if not any(enabled_libs.values()):
        print("\nError: No libraries available for benchmarking!")
        print("Install: pip install diskannpy")
        sys.exit(1)
    
    # Set data directory based on dataset
    if args.use_sift:
        data_dir = args.data_dir if args.data_dir != './deep10M' else './sift1m'
    else:
        data_dir = args.data_dir
    
    # Load data
    base_vectors, query_vectors, groundtruth = load_deep10m_data(
        data_dir, 
        max_vectors=args.max_vectors,
        use_sift=args.use_sift
    )
    
    # Benchmark parameters
    params = {
        'R': args.R,
        'L_build': args.L_build,
        'L_search': args.L_search,
        'alpha': args.alpha,
        'beam_width': args.beam_width,
        'num_threads': args.num_threads
    }
    
    # Run benchmarks
    results = []
    
    if not args.batch_only:
        # Single query benchmarks
        if enabled_libs['caliby']:
            try:
                result = benchmark_caliby_diskann(base_vectors, query_vectors, groundtruth, params, data_dir)
                results.append(result)
            except Exception as e:
                print(f"\n✗ Caliby DiskANN benchmark failed: {e}")
                import traceback
                traceback.print_exc()
        
        if enabled_libs['diskannpy']:
            try:
                result = benchmark_diskannpy(base_vectors, query_vectors, groundtruth, params, data_dir)
                results.append(result)
            except Exception as e:
                print(f"\n✗ diskannpy benchmark failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Batch search benchmarks
    if enabled_libs['caliby']:
        try:
            result = benchmark_caliby_batch(base_vectors, query_vectors, groundtruth, params, data_dir)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Caliby batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    if enabled_libs['diskannpy']:
        try:
            result = benchmark_diskannpy_batch(base_vectors, query_vectors, groundtruth, params, data_dir)
            results.append(result)
        except Exception as e:
            print(f"\n✗ diskannpy batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    if results:
        print_results_table(results)
    else:
        print("\n✗ All benchmarks failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
