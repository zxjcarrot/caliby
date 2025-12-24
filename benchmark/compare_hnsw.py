#!/usr/bin/env python3
"""
HNSW Benchmark: Compare Caliby vs Usearch vs Faiss

This benchmark compares three popular HNSW implementations:
- Caliby: Buffer-managed HNSW with disk persistence
- Usearch: Fast single-file vector search
- Faiss: Facebook's vector search library

Metrics measured:
- Index build time
- Index size (memory/disk)
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
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import argparse

# Add parent directory to path to import local caliby build
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing all libraries
LIBS_AVAILABLE = {}

try:
    import caliby
    caliby.set_buffer_config(size_gb=3)
    LIBS_AVAILABLE['caliby'] = True
except ImportError:
    print("Warning: caliby not available")
    LIBS_AVAILABLE['caliby'] = False

try:
    import usearch
    from usearch.index import Index as UsearchIndex
    LIBS_AVAILABLE['usearch'] = True
except ImportError:
    print("Warning: usearch not available (install: pip install usearch)")
    LIBS_AVAILABLE['usearch'] = False

try:
    import faiss
    LIBS_AVAILABLE['faiss'] = True
except ImportError:
    print("Warning: faiss not available (install: pip install faiss-cpu)")
    LIBS_AVAILABLE['faiss'] = False


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
    params: Dict


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


def download_sift1m(data_dir='./sift1m'):
    """Download and extract SIFT1M dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    
    base_file = os.path.join(data_dir, 'sift_base.fvecs')
    query_file = os.path.join(data_dir, 'sift_query.fvecs')
    groundtruth_file = os.path.join(data_dir, 'sift_groundtruth.ivecs')
    
    if os.path.exists(base_file) and os.path.exists(query_file) and os.path.exists(groundtruth_file):
        print(f"✓ SIFT1M dataset already exists in {data_dir}")
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
        
        sift_dir = os.path.join(data_dir, 'sift')
        if os.path.exists(sift_dir):
            for filename in os.listdir(sift_dir):
                src = os.path.join(sift_dir, filename)
                dst = os.path.join(data_dir, filename)
                os.rename(src, dst)
            os.rmdir(sift_dir)
        
        os.remove(tar_path)
        print("✓ Download complete")
        return data_dir
        
    except Exception as e:
        print(f"Error downloading SIFT1M: {e}")
        print("Please download manually from: ftp://ftp.irisa.fr/local/texmex/corpus/")
        sys.exit(1)


def load_sift1m_data(data_dir='./sift1m'):
    """Load SIFT1M dataset."""
    print("\nLoading SIFT1M dataset...")
    base_vectors = read_fvecs(os.path.join(data_dir, 'sift_base.fvecs'))
    query_vectors = read_fvecs(os.path.join(data_dir, 'sift_query.fvecs'))
    groundtruth = read_ivecs(os.path.join(data_dir, 'sift_groundtruth.ivecs'))
    
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


def benchmark_caliby(base_vectors, query_vectors, groundtruth, params):
    """Benchmark Caliby HNSW implementation."""
    print("\n" + "="*70)
    print("Benchmarking Caliby HNSW")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  M={params['M']}, ef_construction={params['ef_construction']}")
    
    start_time = time.time()
    index = caliby.HnswIndex(
        max_elements=num_vectors,
        dim=dim,
        M=params['M'],
        ef_construction=params['ef_construction'],
        enable_prefetch=True,
        skip_recovery=False,
        index_id=1,
        name="caliby_hnsw_benchmark"
    )
    
    # Add all vectors using the parallel implementation
    print(f"  Adding {num_vectors} vectors...")
    index.add_points(base_vectors, num_threads=0)
    
    build_time = time.time() - start_time
    print(f"✓ Build time: {build_time:.2f}s")
    
    # Flush to get accurate size
    index.flush()
    
    # Estimate index size (BufferManager memory usage)
    # Note: Caliby uses buffer-managed approach, size depends on buffer pool
    index_size_mb = num_vectors * dim * 4 / (1024 * 1024)  # Rough estimate
    print(f"✓ Estimated index size: {index_size_mb:.2f} MB")
    
    # Warm-up
    print("\nWarming up...")
    for i in range(min(100, num_queries)):
        _ = index.search_knn(query_vectors[i], k=10, ef_search=params['ef_search'])
    
    # Benchmark search
    print(f"Running search benchmark (ef_search={params['ef_search']})...")
    latencies = []
    all_results = []
    
    for i in range(num_queries):
        start = time.perf_counter()
        labels, distances = index.search_knn(query_vectors[i], k=10, ef_search=params['ef_search'])
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(labels)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries", end='\r')
    
    print(f"  Processed {num_queries}/{num_queries} queries")
    
    # Calculate metrics
    latencies = np.array(latencies)
    results_array = np.array(all_results)
    qps = num_queries / (np.sum(latencies) / 1000)
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    
    return BenchmarkResult(
        library='Caliby',
        build_time=build_time,
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        recall_at_10=recall,
        params=params
    )



def benchmark_usearch(base_vectors, query_vectors, groundtruth, params):
    """Benchmark Usearch HNSW implementation."""
    print("\n" + "="*70)
    print("Benchmarking Usearch HNSW")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  M={params['M']}, ef_construction={params['ef_construction']}")
    
    start_time = time.time()
    index = UsearchIndex(
        ndim=dim,
        metric='l2sq',
        dtype='f32',
        connectivity=params['M'],
        expansion_add=params['ef_construction'],
        expansion_search=params['ef_search']
    )
    
    # Add vectors with IDs
    ids = np.arange(num_vectors, dtype=np.int64)
    index.add(ids, base_vectors)
    
    build_time = time.time() - start_time
    print(f"✓ Build time: {build_time:.2f}s")
    
    # Get index size (in-memory)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "usearch.index")
        index.save(temp_file)
        index_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
    print(f"✓ Index size: {index_size_mb:.2f} MB")
    
    # Warm-up
    print("\nWarming up...")
    for i in range(min(100, num_queries)):
        _ = index.search(query_vectors[i], 10)
    
    # Benchmark search
    print(f"Running search benchmark (ef_search={params['ef_search']})...")
    latencies = []
    all_results = []
    
    for i in range(num_queries):
        start = time.perf_counter()
        matches = index.search(query_vectors[i], 10)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(matches.keys.astype(np.int32))
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries", end='\r')
    
    print(f"  Processed {num_queries}/{num_queries} queries")
    
    # Calculate metrics
    latencies = np.array(latencies)
    results_array = np.array(all_results)
    qps = num_queries / (np.sum(latencies) / 1000)
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    
    return BenchmarkResult(
        library='Usearch',
        build_time=build_time,
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        recall_at_10=recall,
        params=params
    )


def benchmark_faiss(base_vectors, query_vectors, groundtruth, params):
    """Benchmark Faiss HNSW implementation."""
    print("\n" + "="*70)
    print("Benchmarking Faiss HNSW (IndexHNSWFlat)")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    num_queries = query_vectors.shape[0]
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  M={params['M']}, ef_construction={params['ef_construction']}")
    
    start_time = time.time()
    index = faiss.IndexHNSWFlat(dim, params['M'])
    index.hnsw.efConstruction = params['ef_construction']
    index.hnsw.efSearch = params['ef_search']
    
    # Add vectors
    index.add(base_vectors)
    
    build_time = time.time() - start_time
    print(f"✓ Build time: {build_time:.2f}s")
    
    # Get index size (serialize to estimate)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "faiss.index")
        faiss.write_index(index, temp_file)
        index_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
    print(f"✓ Index size: {index_size_mb:.2f} MB")
    
    # Warm-up
    print("\nWarming up...")
    for i in range(min(100, num_queries)):
        _ = index.search(query_vectors[i:i+1], 10)
    
    # Benchmark search
    print(f"Running search benchmark (ef_search={params['ef_search']})...")
    latencies = []
    all_results = []
    
    for i in range(num_queries):
        start = time.perf_counter()
        distances, indices = index.search(query_vectors[i:i+1], 10)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(indices[0])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries", end='\r')
    
    print(f"  Processed {num_queries}/{num_queries} queries")
    
    # Calculate metrics
    latencies = np.array(latencies)
    results_array = np.array(all_results)
    qps = num_queries / (np.sum(latencies) / 1000)
    recall = compute_recall_at_k(results_array, groundtruth, k=10)
    
    return BenchmarkResult(
        library='Faiss',
        build_time=build_time,
        index_size_mb=index_size_mb,
        qps=qps,
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        recall_at_10=recall,
        params=params
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print comparison table of benchmark results."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Library':<15} {'Build(s)':<12} {'Size(MB)':<12} {'QPS':<12} "
          f"{'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Recall@10':<12}")
    print("-" * 100)
    
    # Sort by QPS (descending)
    results_sorted = sorted(results, key=lambda x: x.qps, reverse=True)
    
    for r in results_sorted:
        print(f"{r.library:<15} {r.build_time:<12.2f} {r.index_size_mb:<12.2f} "
              f"{r.qps:<12.1f} {r.latency_p50:<10.3f} {r.latency_p95:<10.3f} "
              f"{r.latency_p99:<10.3f} {r.recall_at_10:<12.4f}")
    
    print("\n" + "="*100)
    
    # Find best in each category
    best_qps = max(results, key=lambda x: x.qps)
    best_recall = max(results, key=lambda x: x.recall_at_10)
    best_latency = min(results, key=lambda x: x.latency_p50)
    smallest_index = min(results, key=lambda x: x.index_size_mb)
    fastest_build = min(results, key=lambda x: x.build_time)
    
    print("\nWINNERS:")
    print(f"  🏆 Highest QPS:        {best_qps.library} ({best_qps.qps:.1f} queries/sec)")
    print(f"  🏆 Best Recall@10:     {best_recall.library} ({best_recall.recall_at_10:.4f})")
    print(f"  🏆 Lowest P50 Latency: {best_latency.library} ({best_latency.latency_p50:.3f} ms)")
    print(f"  🏆 Smallest Index:     {smallest_index.library} ({smallest_index.index_size_mb:.2f} MB)")
    print(f"  🏆 Fastest Build:      {fastest_build.library} ({fastest_build.build_time:.2f} s)")
    print("\n" + "="*100)


def main():
    # print caliby.__version__
    
    print(f"Caliby version: {caliby.__version__}")
    
    parser = argparse.ArgumentParser(description='HNSW Benchmark: Caliby vs Usearch vs Faiss')
    parser.add_argument('--data-dir', default='./sift1m', help='Directory for SIFT1M dataset')
    parser.add_argument('--M', type=int, default=16, help='HNSW M parameter (connectivity)')
    parser.add_argument('--ef-construction', type=int, default=100, help='ef_construction parameter')
    parser.add_argument('--ef-search', type=int, default=50, help='ef_search parameter')
    parser.add_argument('--skip', type=str,
                        help='Libraries to skip (comma or space separated, e.g., "usearch,faiss" or "usearch faiss")')
    args = parser.parse_args()
    
    print("="*100)
    print("HNSW BENCHMARK: Caliby vs Usearch vs Faiss")
    print("="*100)
    print(f"\nDataset: SIFT1M (1M vectors, 128 dimensions)")
    print(f"Parameters: M={args.M}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    
    # Check available libraries - parse skip argument to support both comma and space separated
    skip_libs = set()
    if args.skip:
        # Split by comma first, then by space
        for part in args.skip.replace(',', ' ').split():
            lib = part.strip().lower()
            if lib in ['caliby', 'usearch', 'faiss']:
                skip_libs.add(lib)
            elif lib:  # non-empty but invalid
                print(f"Warning: Unknown library '{lib}' in --skip, ignoring. Valid options: caliby, usearch, faiss")
    enabled_libs = {lib: available and lib not in skip_libs 
                    for lib, available in LIBS_AVAILABLE.items()}
    
    print(f"\nLibraries enabled:")
    for lib, enabled in enabled_libs.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {lib:<10} {status}")
    
    if not any(enabled_libs.values()):
        print("\nError: No libraries available for benchmarking!")
        print("Install at least one: pip install usearch faiss-cpu")
        sys.exit(1)
    
    # Download and load data
    download_sift1m(args.data_dir)
    base_vectors, query_vectors, groundtruth = load_sift1m_data(args.data_dir)
    
    # Benchmark parameters
    params = {
        'M': args.M,
        'ef_construction': args.ef_construction,
        'ef_search': args.ef_search
    }
    
    # Run benchmarks
    results = []
    
    if enabled_libs['caliby']:
        try:
            result = benchmark_caliby(base_vectors, query_vectors, groundtruth, params)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Caliby benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    if enabled_libs['usearch']:
        try:
            result = benchmark_usearch(base_vectors, query_vectors, groundtruth, params)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Usearch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    if enabled_libs['faiss']:
        try:
            result = benchmark_faiss(base_vectors, query_vectors, groundtruth, params)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Faiss benchmark failed: {e}")
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
