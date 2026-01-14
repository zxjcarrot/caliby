#!/usr/bin/env python3
"""
Larger-than-Memory Workload Benchmark: Caliby IVF+PQ vs FAISS IVF+PQ with mmap

This benchmark compares Caliby's buffer-managed IVF+PQ approach against FAISS with mmap (IO_FLAG_MMAP)
for workloads that exceed available physical memory. Uses SIFT1M dataset.

Test scenarios:
- Limited physical memory (simulated via buffer pool size for Caliby, mmap for FAISS)
- Cold cache performance
- Memory efficiency
- Search performance under memory pressure

Metrics:
- Index build time
- Index size (disk)
- Memory usage (RSS)
- Search throughput (QPS)
- Search latency (P50, P95, P99)
- Recall@10 accuracy
"""

import numpy as np
import time
import sys
import os
import struct
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import argparse

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available")

# Add caliby to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import caliby

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: FAISS not installed. Install with: pip install faiss-cpu")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    library: str
    memory_limit_mb: int  # Simulated memory limit
    num_clusters: int = 256
    num_subquantizers: int = 8
    nprobe: int = 8
    k: int = 10
    num_threads: int = 1
    warmup_queries: int = 100


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    library: str
    config: str
    memory_limit_mb: int
    
    # Build metrics
    build_time: float = 0.0
    train_time: float = 0.0
    add_time: float = 0.0
    index_size_mb: float = 0.0
    rss_after_build_mb: float = 0.0
    
    # Search metrics - cold cache
    cold_qps: float = 0.0
    cold_latency_p50: float = 0.0
    cold_latency_p95: float = 0.0
    cold_latency_p99: float = 0.0
    cold_recall: float = 0.0
    rss_after_cold_mb: float = 0.0
    
    # Search metrics - warm cache
    warm_qps: float = 0.0
    warm_latency_p50: float = 0.0
    warm_latency_p95: float = 0.0
    warm_latency_p99: float = 0.0
    warm_recall: float = 0.0
    rss_after_warm_mb: float = 0.0
    
    params: Dict = field(default_factory=dict)


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


def read_fbin(filename):
    """Read .fbin file format (used by DEEP dataset)."""
    with open(filename, 'rb') as f:
        # Read header: num_vectors (4 bytes), dimension (4 bytes)
        num_vectors = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # Read all vectors at once
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        return data.reshape(num_vectors, dim)


def read_fvecs_batch(filename, batch_size=10000):
    """Read .fvecs file in batches (streaming). Yields batches of vectors."""
    with open(filename, 'rb') as f:
        batch = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                if batch:
                    yield np.array(batch, dtype=np.float32)
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            batch.append(vec)
            
            if len(batch) >= batch_size:
                yield np.array(batch, dtype=np.float32)
                batch = []


def read_fbin_batch(filename, batch_size=10000):
    """Read .fbin file in batches (streaming). Yields batches of vectors."""
    with open(filename, 'rb') as f:
        # Read header
        num_vectors = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # Read in batches
        vectors_read = 0
        while vectors_read < num_vectors:
            current_batch_size = min(batch_size, num_vectors - vectors_read)
            data = np.fromfile(f, dtype=np.float32, count=current_batch_size * dim)
            batch = data.reshape(current_batch_size, dim)
            vectors_read += current_batch_size
            yield batch


def count_fvecs(filename):
    """Count number of vectors in .fvecs file without loading all into memory."""
    count = 0
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            f.seek(4 * dim, 1)  # Skip vector data
            count += 1
    return count


def count_fbin(filename):
    """Count number of vectors in .fbin file."""
    with open(filename, 'rb') as f:
        num_vectors = struct.unpack('i', f.read(4))[0]
        return num_vectors


def get_fvecs_dim(filename):
    """Get dimension of vectors in .fvecs file."""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError(f"Empty file: {filename}")
        return struct.unpack('i', dim_bytes)[0]


def get_fbin_dim(filename):
    """Get dimension of vectors in .fbin file."""
    with open(filename, 'rb') as f:
        f.read(4)  # Skip num_vectors
        dim = struct.unpack('i', f.read(4))[0]
        return dim


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


def read_ibin(filename):
    """Read .ibin file format (used for ground truth in DEEP dataset)."""
    with open(filename, 'rb') as f:
        # Read header: num_vectors (4 bytes), dimension (4 bytes)
        num_vectors = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        
        # Read all vectors at once
        data = np.fromfile(f, dtype=np.int32, count=num_vectors * dim)
        return data.reshape(num_vectors, dim)


def compute_recall_at_k(results: np.ndarray, groundtruth: np.ndarray, k: int = 10) -> float:
    """Compute Recall@k for search results."""
    n_queries = results.shape[0]
    recalls = []
    
    for i in range(n_queries):
        result_set = set(results[i, :k])
        gt_set = set(groundtruth[i, :k])
        recall = len(result_set & gt_set) / k
        recalls.append(recall)
    
    return np.mean(recalls)


def get_rss_mb():
    """Get current process RSS in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def get_file_size_mb(filepath):
    """Get file size in MB."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def evict_file_from_cache(filepath):
    """Evict file from OS page cache using vmtouch."""
    try:
        subprocess.run(['vmtouch', '-e', filepath], check=True, capture_output=True)
        print(f"  ✓ Evicted {filepath} from page cache")
    except Exception as e:
        print(f"  Warning: Could not evict file (is vmtouch installed?): {e}")


def detect_file_format(filename):
    """Detect if file is .fvecs/.ivecs or .fbin/.ibin format."""
    if filename.endswith('.fvecs') or filename.endswith('.ivecs'):
        return 'vecs'
    elif filename.endswith('.fbin') or filename.endswith('.ibin'):
        return 'bin'
    else:
        raise ValueError(f"Unknown file format: {filename}")


def read_vectors(filename):
    """Auto-detect format and read vectors."""
    fmt = detect_file_format(filename)
    if fmt == 'vecs':
        if filename.endswith('.ivecs'):
            return read_ivecs(filename)
        else:
            return read_fvecs(filename)
    elif fmt == 'bin':
        if filename.endswith('.ibin'):
            return read_ibin(filename)
        else:
            return read_fbin(filename)


def read_vectors_batch(filename, batch_size=10000):
    """Auto-detect format and read vectors in batches."""
    fmt = detect_file_format(filename)
    if fmt == 'vecs':
        return read_fvecs_batch(filename, batch_size)
    elif fmt == 'bin':
        return read_fbin_batch(filename, batch_size)


def count_vectors(filename):
    """Auto-detect format and count vectors."""
    fmt = detect_file_format(filename)
    if fmt == 'vecs':
        return count_fvecs(filename)
    elif fmt == 'bin':
        return count_fbin(filename)


def get_vector_dim(filename):
    """Auto-detect format and get vector dimension."""
    fmt = detect_file_format(filename)
    if fmt == 'vecs':
        return get_fvecs_dim(filename)
    elif fmt == 'bin':
        return get_fbin_dim(filename)


def is_cgroup_v2():
    """Check if system uses cgroup v2."""
    return os.path.exists('/sys/fs/cgroup/cgroup.controllers')


def run_with_memory_limit(func, memory_limit_mb, *args, **kwargs):
    """Run function under cgroup memory limit to enforce total memory usage including page cache."""
    cgroup_name = f"ivfpq_benchmark"
    current_pid = os.getpid()
    cgroup_path = None
    
    try:
        # Try cgroup v2 first
        if is_cgroup_v2():
            print(f"  Attempting to use cgroup v2 with MemoryMax={memory_limit_mb}MB")
            cgroup_path = f"/sys/fs/cgroup/{cgroup_name}"
            
            try:
                # Create cgroup directory
                subprocess.run(['sudo', 'mkdir', '-p', cgroup_path], check=True, capture_output=True)
                
                # Set memory limit
                memory_limit_bytes = memory_limit_mb * 1024 * 1024
                subprocess.run(['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {cgroup_path}/memory.max'], 
                              check=True, capture_output=True)
                
                # Move current process into cgroup
                subprocess.run(['sudo', 'sh', '-c', f'echo {current_pid} > {cgroup_path}/cgroup.procs'], 
                              check=True, capture_output=True)
                
                print(f"  ✓ Process {current_pid} moved to cgroup v2: {cgroup_path}")
                print(f"  ✓ Memory limit enforced: {memory_limit_mb}MB")
                
                # Run the function
                result = func(*args, **kwargs)
                return result
                
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  Failed to setup cgroup v2: {e}")
                print(f"  ⚠️  Trying cgroup v1...")
                cgroup_path = None
        
        # Try cgroup v1
        if not cgroup_path:
            print(f"  Attempting to use cgroup v1 with memory limit={memory_limit_mb}MB")
            cgroup_path = f"/sys/fs/cgroup/memory/{cgroup_name}"
            
            try:
                # Create cgroup directory
                subprocess.run(['sudo', 'mkdir', '-p', cgroup_path], check=True, capture_output=True)
                
                # Set memory limit
                memory_limit_bytes = memory_limit_mb * 1024 * 1024
                subprocess.run(['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {cgroup_path}/memory.limit_in_bytes'], 
                              check=True, capture_output=True)
                
                # Set swap limit (disable swap or set to same as memory)
                subprocess.run(['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {cgroup_path}/memory.memsw.limit_in_bytes'], 
                              check=True, stderr=subprocess.DEVNULL, capture_output=True)
                
                # Move current process into cgroup
                subprocess.run(['sudo', 'sh', '-c', f'echo {current_pid} > {cgroup_path}/tasks'], 
                              check=True, capture_output=True)
                
                print(f"  ✓ Process {current_pid} moved to cgroup v1: {cgroup_path}")
                print(f"  ✓ Memory limit enforced: {memory_limit_mb}MB")
                
                # Run the function
                result = func(*args, **kwargs)
                return result
                
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  Failed to setup cgroup v1: {e}")
                cgroup_path = None
        
        # If both failed, warn and run without limit
        if not cgroup_path:
            print("  ⚠️  WARNING: Could not setup cgroup!")
            print("  ⚠️  Running without memory limit - results may not reflect true OOM behavior")
            print("  ⚠️  Make sure you have sudo access and cgroups are enabled")
        
        # Run without cgroup
        return func(*args, **kwargs)
        
    finally:
        # Clean up cgroup
        if cgroup_path and os.path.exists(cgroup_path):
            try:
                subprocess.run(['sudo', 'rmdir', cgroup_path], check=True, capture_output=True)
            except:
                pass




def benchmark_caliby_ivfpq_oom(base_file, query_vectors, groundtruth, config: BenchmarkConfig,
                                data_dir: str, batch_size: int = 10000) -> BenchmarkResult:
    """
    Benchmark Caliby IVF+PQ with constrained buffer pool size using batch-based inserts.
    
    Args:
        base_file: Path to base dataset file (.fvecs or .fbin)
        query_vectors: Query vectors
        groundtruth: Ground truth nearest neighbors
        config: Benchmark configuration
        data_dir: Directory containing dataset files
        batch_size: Number of vectors to insert per batch (default: 10000)
    """
    print("\n" + "="*80)
    print(f"Benchmarking Caliby IVF+PQ (Buffer Limit: {config.memory_limit_mb}MB)")
    print("="*80)
    
    # Get dataset info without loading all vectors (auto-detect format)
    num_vectors = count_vectors(base_file)
    dim = get_vector_dim(base_file)
    num_queries = query_vectors.shape[0]
    
    print(f"Dataset: {num_vectors} vectors, dim={dim}")
    print(f"Using batch-based inserts: {batch_size} vectors per batch")
    
    result = BenchmarkResult(
        library='caliby',
        config=f'buffer_{config.memory_limit_mb}mb',
        memory_limit_mb=config.memory_limit_mb,
        params={
            'num_clusters': config.num_clusters,
            'num_subquantizers': config.num_subquantizers,
            'nprobe': config.nprobe,
            'num_threads': config.num_threads
        }
    )
    
    # Setup temporary directory
    # Create a unique temporary directory for this run to avoid any recovery issues
    tmpdir = tempfile.mkdtemp(prefix='caliby_ivfpq_oom_')
    
    # Ensure clean state - delete directory if it exists (shouldn't happen with mkdtemp but be safe)
    if os.path.exists(tmpdir):
        import shutil
        shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)
    
    # Configure buffer pool size (convert MB to GB for set_buffer_config)
    buffer_size_gb = config.memory_limit_mb / 1024.0
    caliby.set_buffer_config(size_gb=buffer_size_gb)
    caliby.open(tmpdir, cleanup_if_exist=True)
    print(f"  Buffer pool size: {buffer_size_gb:.2f} GB")
    print(f"  Index directory: {tmpdir}")
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  num_clusters={config.num_clusters}, num_subquantizers={config.num_subquantizers}")
    
    rss_before = get_rss_mb()
    start_time = time.time()
    
    index = caliby.IVFPQIndex(
        max_elements=num_vectors + 10000,
        dim=dim,
        num_clusters=config.num_clusters,
        num_subquantizers=config.num_subquantizers,
        skip_recovery=True,  # Force fresh index creation, don't attempt recovery
        index_id=1,
        name="caliby_ivfpq_oom"
    )
    
    # Train on first batch (only if not already trained from recovery)
    if not index.is_trained():
        print("  Training on first batch...")
        train_start = time.time()
        first_batch = next(read_vectors_batch(base_file, min(500000, num_vectors)))
        index.train(first_batch)
        result.train_time = time.time() - train_start
    else:
        print("  Index already trained (recovered from previous run)")
        result.train_time = 0.0
    
    # Add vectors in batches (streaming from disk)
    print(f"  Adding {num_vectors} vectors in batches of {batch_size}...")
    add_start = time.time()
    total_inserted = 0
    
    for batch_idx, batch_vectors in enumerate(read_vectors_batch(base_file, batch_size)):
        index.add_points(batch_vectors)
        total_inserted += len(batch_vectors)
        if (total_inserted % 50000) == 0 or batch_idx % 10 == 0:
            print(f"    Progress: {total_inserted}/{num_vectors} ({total_inserted/num_vectors*100:.1f}%)")
    
    result.add_time = time.time() - add_start
    print(f"  ✓ Inserted {total_inserted} vectors")
    
    result.build_time = time.time() - start_time
    result.rss_after_build_mb = get_rss_mb()
    
    print(f"✓ Build time: {result.build_time:.2f}s (train={result.train_time:.2f}s, add={result.add_time:.2f}s)")
    print(f"✓ RSS: {rss_before:.1f} MB -> {result.rss_after_build_mb:.1f} MB")
    
    # Get index size from heapfile
    heapfile_path = "heapfile"
    if os.path.exists(heapfile_path):
        result.index_size_mb = get_file_size_mb(heapfile_path)
        print(f"✓ Index size: {result.index_size_mb:.2f} MB")
    
    # Cold cache search - clear buffer and evict heapfile
    print("\n--- Cold Cache Search ---")
    print("Clearing buffer pool and evicting index from OS cache...")
    
    # Delete and recreate index to clear buffer pool
    del index
    caliby.close()
    time.sleep(1)
    
    # Evict heapfile from OS cache
    if os.path.exists(heapfile_path):
        evict_file_from_cache(heapfile_path)
    
    # Reopen and recreate index (will load from disk on-demand)
    caliby.open(tmpdir, cleanup_if_exist=False)
    index = caliby.IVFPQIndex(
        max_elements=num_vectors + 10000,
        dim=dim,
        num_clusters=config.num_clusters,
        num_subquantizers=config.num_subquantizers,
        index_id=1,
        name="caliby_ivfpq_oom",
        skip_recovery=False  # Enable recovery of existing index
    )
    
    # Define cold search function to run under memory limit
    def cold_search_func():
        print(f"\nCold search (nprobe={config.nprobe})...")
        latencies = []
        all_results = []
        
        start_time = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            labels, distances = index.search_knn(query_vectors[i], k=config.k, nprobe=config.nprobe, stats=False)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            # Ensure labels is a numpy array and pad if necessary
            labels = np.array(labels)
            if len(labels) < config.k:
                labels = np.pad(labels, (0, config.k - len(labels)), constant_values=-1)
            all_results.append(labels[:config.k])
        
        qps = num_queries / (time.time() - start_time)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
        rss = get_rss_mb()
        
        return qps, latency_p50, latency_p95, latency_p99, recall, rss
    
    # Run cold search under memory limit
    print(f"\nEnforcing memory limit during cold cache search...")
    (result.cold_qps, result.cold_latency_p50, result.cold_latency_p95, 
     result.cold_latency_p99, result.cold_recall, result.rss_after_cold_mb) = run_with_memory_limit(
        cold_search_func, config.memory_limit_mb
    )
    
    print(f"✓ Cold QPS: {result.cold_qps:.2f}")
    print(f"✓ Cold Latency: P50={result.cold_latency_p50:.3f}ms, P95={result.cold_latency_p95:.3f}ms, P99={result.cold_latency_p99:.3f}ms")
    print(f"✓ Cold Recall@{config.k}: {result.cold_recall:.4f}")
    print(f"✓ RSS after cold: {result.rss_after_cold_mb:.1f} MB")
    
    # Warm cache search
    print("\n--- Warm Cache Search ---")
    print(f"Running {config.warmup_queries} warmup queries...")
    for i in range(min(config.warmup_queries, num_queries)):
        _ = index.search_knn(query_vectors[i], k=config.k, nprobe=config.nprobe, stats=False)
    
    # Define warm search function to run under memory limit
    def warm_search_func():
        print(f"\nWarm search (nprobe={config.nprobe})...")
        latencies = []
        all_results = []
        
        start_time = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            labels, distances = index.search_knn(query_vectors[i], k=config.k, nprobe=config.nprobe, stats=False)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            # Ensure labels is a numpy array and pad if necessary
            labels = np.array(labels)
            if len(labels) < config.k:
                labels = np.pad(labels, (0, config.k - len(labels)), constant_values=-1)
            all_results.append(labels[:config.k])
        
        qps = num_queries / (time.time() - start_time)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
        rss = get_rss_mb()
        
        return qps, latency_p50, latency_p95, latency_p99, recall, rss
    
    # Run warm search under memory limit
    (result.warm_qps, result.warm_latency_p50, result.warm_latency_p95,
     result.warm_latency_p99, result.warm_recall, result.rss_after_warm_mb) = run_with_memory_limit(
        warm_search_func, config.memory_limit_mb
    )
    
    print(f"✓ Warm QPS: {result.warm_qps:.2f}")
    print(f"✓ Warm Latency: P50={result.warm_latency_p50:.3f}ms, P95={result.warm_latency_p95:.3f}ms, P99={result.warm_latency_p99:.3f}ms")
    print(f"✓ Warm Recall@{config.k}: {result.warm_recall:.4f}")
    print(f"✓ RSS after warm: {result.rss_after_warm_mb:.1f} MB")
    
    # Cleanup
    caliby.close()
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    
    return result


def benchmark_faiss_ivfpq_mmap(base_file, query_vectors, groundtruth, config: BenchmarkConfig,
                                 data_dir: str, index_file: Optional[str] = None, batch_size: int = 10000) -> BenchmarkResult:
    """
    Benchmark FAISS IVF+PQ with mmap (IO_FLAG_MMAP) for larger-than-memory workload.
    
    Args:
        base_file: Path to base dataset file (.fvecs)
        query_vectors: Query vectors
        groundtruth: Ground truth nearest neighbors
        config: Benchmark configuration
        data_dir: Directory containing dataset files
        index_file: Path to saved index file (will create if None)
        batch_size: Number of vectors to insert per batch (default: 10000)
    """
    if not HAS_FAISS:
        print("FAISS not available, skipping")
        return None
    
    print("\n" + "="*80)
    print(f"Benchmarking FAISS IVF+PQ with mmap (Memory Limit: {config.memory_limit_mb}MB)")
    print("="*80)
    
    # Get dataset info without loading all vectors (auto-detect format)
    num_vectors = count_vectors(base_file)
    dim = get_vector_dim(base_file)
    num_queries = query_vectors.shape[0]
    
    print(f"Dataset: {num_vectors} vectors, dim={dim}")
    print(f"Using batch-based inserts: {batch_size} vectors per batch")
    
    result = BenchmarkResult(
        library='faiss',
        config=f'mmap_{config.memory_limit_mb}mb',
        memory_limit_mb=config.memory_limit_mb,
        params={
            'num_clusters': config.num_clusters,
            'num_subquantizers': config.num_subquantizers,
            'nprobe': config.nprobe,
            'num_threads': config.num_threads
        }
    )
    
    # Use temporary file if index_file not provided
    if index_file is None:
        index_file = os.path.join(data_dir, 'faiss_ivfpq_oom.index')
    # remove index_file if exists
    if os.path.exists(index_file):
        os.remove(index_file)
    # Build or load index
    if not os.path.exists(index_file):
        print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
        print(f"  num_clusters={config.num_clusters}, num_subquantizers={config.num_subquantizers}")
        
        rss_before = get_rss_mb()
        start_time = time.time()
        
        # Create IVF+PQ index
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            config.num_clusters,
            config.num_subquantizers,
            8  # nbits per subquantizer
        )
        
        # Train on first batch
        print("  Training on first batch...")
        train_start = time.time()
        first_batch = next(read_vectors_batch(base_file, min(500000, num_vectors)))
        index.train(first_batch)
        result.train_time = time.time() - train_start
        
        # Add vectors in batches (streaming from disk)
        print(f"  Adding {num_vectors} vectors in batches of {batch_size}...")
        add_start = time.time()
        total_inserted = 0
        
        for batch_idx, batch_vectors in enumerate(read_vectors_batch(base_file, batch_size)):
            index.add(batch_vectors)
            total_inserted += len(batch_vectors)
            if (total_inserted % 50000) == 0 or batch_idx % 10 == 0:
                print(f"    Progress: {total_inserted}/{num_vectors} ({total_inserted/num_vectors*100:.1f}%)")
        
        result.add_time = time.time() - add_start
        print(f"  ✓ Inserted {total_inserted} vectors")
        
        result.build_time = time.time() - start_time
        result.rss_after_build_mb = get_rss_mb()
        
        print(f"✓ Build time: {result.build_time:.2f}s (train={result.train_time:.2f}s, add={result.add_time:.2f}s)")
        print(f"✓ RSS: {rss_before:.1f} MB -> {result.rss_after_build_mb:.1f} MB")
        
        # Save index
        print(f"\nSaving index to {index_file}...")
        save_start = time.time()
        faiss.write_index(index, index_file)
        save_time = time.time() - save_start
        print(f"✓ Saved in {save_time:.2f}s")
        
        del index
        time.sleep(1)
    else:
        print(f"Using existing index: {index_file}")
        result.build_time = 0.0  # Already built
    
    result.index_size_mb = get_file_size_mb(index_file)
    print(f"✓ Index size: {result.index_size_mb:.2f} MB")
    
    # Cold cache search - evict index from OS cache and load with mmap
    print("\n--- Cold Cache Search ---")
    print("Evicting index from OS cache...")
    evict_file_from_cache(index_file)
    
    # Define cold search function to run under memory limit
    def cold_search_func():
        print(f"Loading index with mmap (IO_FLAG_MMAP)...")
        load_start = time.time()
        index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
        load_time = time.time() - load_start
        rss_after_load = get_rss_mb()
        print(f"✓ Loaded in {load_time:.4f}s (RSS: {rss_after_load:.1f} MB)")
        
        # Configure nprobe
        index.nprobe = config.nprobe
        faiss.omp_set_num_threads(config.num_threads)
        
        print(f"\nCold search (nprobe={config.nprobe})...")
        latencies = []
        all_results = []
        
        start_time = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            distances, labels = index.search(query_vectors[i].reshape(1, -1), config.k)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            all_results.append(labels[0])
        
        qps = num_queries / (time.time() - start_time)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
        rss = get_rss_mb()
        
        return index, qps, latency_p50, latency_p95, latency_p99, recall, rss
    
    # Run cold search under memory limit
    print(f"\nEnforcing memory limit during cold cache search...")
    (index, result.cold_qps, result.cold_latency_p50, result.cold_latency_p95,
     result.cold_latency_p99, result.cold_recall, result.rss_after_cold_mb) = run_with_memory_limit(
        cold_search_func, config.memory_limit_mb
    )
    
    print(f"✓ Cold QPS: {result.cold_qps:.2f}")
    print(f"✓ Cold Latency: P50={result.cold_latency_p50:.3f}ms, P95={result.cold_latency_p95:.3f}ms, P99={result.cold_latency_p99:.3f}ms")
    print(f"✓ Cold Recall@{config.k}: {result.cold_recall:.4f}")
    print(f"✓ RSS after cold: {result.rss_after_cold_mb:.1f} MB")
    
    # Warm cache search
    print("\n--- Warm Cache Search ---")
    print(f"Running {config.warmup_queries} warmup queries...")
    for i in range(min(config.warmup_queries, num_queries)):
        _ = index.search(query_vectors[i].reshape(1, -1), config.k)
    
    # Define warm search function to run under memory limit
    def warm_search_func():
        print(f"\nWarm search (nprobe={config.nprobe})...")
        latencies = []
        all_results = []
        
        start_time = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            distances, labels = index.search(query_vectors[i].reshape(1, -1), config.k)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            all_results.append(labels[0])
        
        qps = num_queries / (time.time() - start_time)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
        rss = get_rss_mb()
        
        return qps, latency_p50, latency_p95, latency_p99, recall, rss
    
    # Run warm search under memory limit
    (result.warm_qps, result.warm_latency_p50, result.warm_latency_p95,
     result.warm_latency_p99, result.warm_recall, result.rss_after_warm_mb) = run_with_memory_limit(
        warm_search_func, config.memory_limit_mb
    )
    
    print(f"✓ Warm QPS: {result.warm_qps:.2f}")
    print(f"✓ Warm Latency: P50={result.warm_latency_p50:.3f}ms, P95={result.warm_latency_p95:.3f}ms, P99={result.warm_latency_p99:.3f}ms")
    print(f"✓ Warm Recall@{config.k}: {result.warm_recall:.4f}")
    print(f"✓ RSS after warm: {result.rss_after_warm_mb:.1f} MB")
    
    return result


def print_comparison_table(caliby_result: BenchmarkResult, faiss_result: Optional[BenchmarkResult]):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Metric':<40} {'Caliby':<20} {'FAISS+mmap':<20} {'Speedup':<15}")
    print("-" * 100)
    
    # Build metrics
    print("\n--- Build Performance ---")
    print(f"{'Build time (s)':<40} {caliby_result.build_time:<20.2f}", end='')
    if faiss_result:
        print(f"{faiss_result.build_time:<20.2f} ", end='')
        if faiss_result.build_time > 0:
            speedup = faiss_result.build_time / caliby_result.build_time
            print(f"{speedup:.2f}x")
        else:
            print("N/A")
    else:
        print()
    
    print(f"{'  - Train time (s)':<40} {caliby_result.train_time:<20.2f}", end='')
    if faiss_result:
        print(f"{faiss_result.train_time:<20.2f}")
    else:
        print()
    
    print(f"{'  - Add time (s)':<40} {caliby_result.add_time:<20.2f}", end='')
    if faiss_result:
        print(f"{faiss_result.add_time:<20.2f}")
    else:
        print()
    
    print(f"{'Index size (MB)':<40} {caliby_result.index_size_mb:<20.2f}", end='')
    if faiss_result:
        print(f"{faiss_result.index_size_mb:<20.2f}")
    else:
        print()
    
    print(f"{'RSS after build (MB)':<40} {caliby_result.rss_after_build_mb:<20.1f}", end='')
    if faiss_result:
        print(f"{faiss_result.rss_after_build_mb:<20.1f}")
    else:
        print()
    
    # Cold cache metrics
    print("\n--- Cold Cache Performance ---")
    print(f"{'QPS':<40} {caliby_result.cold_qps:<20.2f}", end='')
    if faiss_result:
        print(f"{faiss_result.cold_qps:<20.2f} ", end='')
        speedup = caliby_result.cold_qps / faiss_result.cold_qps
        print(f"{speedup:.2f}x")
    else:
        print()
    
    print(f"{'Latency P50 (ms)':<40} {caliby_result.cold_latency_p50:<20.3f}", end='')
    if faiss_result:
        print(f"{faiss_result.cold_latency_p50:<20.3f}")
    else:
        print()
    
    print(f"{'Latency P95 (ms)':<40} {caliby_result.cold_latency_p95:<20.3f}", end='')
    if faiss_result:
        print(f"{faiss_result.cold_latency_p95:<20.3f}")
    else:
        print()
    
    print(f"{'Latency P99 (ms)':<40} {caliby_result.cold_latency_p99:<20.3f}", end='')
    if faiss_result:
        print(f"{faiss_result.cold_latency_p99:<20.3f}")
    else:
        print()
    
    print(f"{'Recall@10':<40} {caliby_result.cold_recall:<20.4f}", end='')
    if faiss_result:
        print(f"{faiss_result.cold_recall:<20.4f}")
    else:
        print()
    
    print(f"{'RSS (MB)':<40} {caliby_result.rss_after_cold_mb:<20.1f}", end='')
    if faiss_result:
        print(f"{faiss_result.rss_after_cold_mb:<20.1f}")
    else:
        print()
    
    # Warm cache metrics
    print("\n--- Warm Cache Performance ---")
    print(f"{'QPS':<40} {caliby_result.warm_qps:<20.2f}", end='')
    if faiss_result:
        print(f"{faiss_result.warm_qps:<20.2f} ", end='')
        speedup = caliby_result.warm_qps / faiss_result.warm_qps
        print(f"{speedup:.2f}x")
    else:
        print()
    
    print(f"{'Latency P50 (ms)':<40} {caliby_result.warm_latency_p50:<20.3f}", end='')
    if faiss_result:
        print(f"{faiss_result.warm_latency_p50:<20.3f}")
    else:
        print()
    
    print(f"{'Latency P95 (ms)':<40} {caliby_result.warm_latency_p95:<20.3f}", end='')
    if faiss_result:
        print(f"{faiss_result.warm_latency_p95:<20.3f}")
    else:
        print()
    
    print(f"{'Latency P99 (ms)':<40} {caliby_result.warm_latency_p99:<20.3f}", end='')
    if faiss_result:
        print(f"{faiss_result.warm_latency_p99:<20.3f}")
    else:
        print()
    
    print(f"{'Recall@10':<40} {caliby_result.warm_recall:<20.4f}", end='')
    if faiss_result:
        print(f"{faiss_result.warm_recall:<20.4f}")
    else:
        print()
    
    print(f"{'RSS (MB)':<40} {caliby_result.rss_after_warm_mb:<20.1f}", end='')
    if faiss_result:
        print(f"{faiss_result.rss_after_warm_mb:<20.1f}")
    else:
        print()
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Caliby IVF+PQ vs FAISS IVF+PQ with mmap for OOM scenarios')
    parser.add_argument('--memory-limit', type=int, default=4096, help='Memory limit in MB (default: 4096)')
    parser.add_argument('--nprobe', type=int, default=32, help='Number of clusters to probe (default: 32)')
    parser.add_argument('--k', type=int, default=10, help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--data-dir', type=str, default='deep10M', help='Path to dataset directory (default: deep10M)')
    parser.add_argument('--caliby-only', action='store_true', help='Run Caliby benchmark only')
    parser.add_argument('--faiss-only', action='store_true', help='Run FAISS benchmark only')
    parser.add_argument('--reuse-faiss-index', action='store_true', help='Reuse existing FAISS index if available')
    
    args = parser.parse_args()
    
    # Check for dataset
    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    
    # Auto-detect dataset files (support both sift1m and deep10M naming conventions and formats)
    possible_base_files = ['base.10M.fbin', 'deep1B_base.fvecs', 'sift_base.fvecs']
    possible_query_files = ['query.public.10K.fbin', 'deep1B_query.fvecs', 'sift_query.fvecs']
    possible_gt_files = ['groundtruth.ibin', 'deep1B_groundtruth.ivecs', 'sift_groundtruth.ivecs']
    
    base_file = None
    for fname in possible_base_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            base_file = fpath
            break
    
    query_file = None
    for fname in possible_query_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            query_file = fpath
            break
    
    gt_file = None
    for fname in possible_gt_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            gt_file = fpath
            break
    
    if not base_file or not query_file or not gt_file:
        print("ERROR: Dataset not found!")
        print(f"Searched in: {data_dir}/")
        print("Expected files:")
        print("  Base vectors: base.10M.fbin, deep1B_base.fvecs, or sift_base.fvecs")
        print("  Query vectors: query.public.10K.fbin, deep1B_query.fvecs, or sift_query.fvecs")
        print("  Ground truth: groundtruth.ibin, deep1B_groundtruth.ivecs, or sift_groundtruth.ivecs")
        print("\nFor DEEP10M dataset, download from: http://sites.skoltech.ru/compvision/noimi/")
        print("For SIFT1M dataset, download from: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz")
        return 1
    
    print("Loading query vectors and ground truth...")
    query_vectors = read_vectors(query_file)
    groundtruth = read_vectors(gt_file)
    print(f"  Queries: {query_vectors.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    # Create benchmark config
    config = BenchmarkConfig(
        library='caliby',
        memory_limit_mb=args.memory_limit,
        num_clusters=256,
        num_subquantizers=32,
        nprobe=args.nprobe,
        k=args.k,
        num_threads=1,
        warmup_queries=100
    )
    
    # Run benchmarks
    caliby_result = None
    faiss_result = None
    
    if not args.faiss_only:
        caliby_result = benchmark_caliby_ivfpq_oom(
            base_file, query_vectors, groundtruth, config, data_dir
        )
    
    if not args.caliby_only and HAS_FAISS:
        faiss_config = BenchmarkConfig(
            library='faiss',
            memory_limit_mb=args.memory_limit,
            num_clusters=256,
            num_subquantizers=32,
            nprobe=args.nprobe,
            k=args.k,
            num_threads=1,
            warmup_queries=100
        )
        
        faiss_index_file = os.path.join(data_dir, 'faiss_ivfpq_oom.index') if args.reuse_faiss_index else None
        faiss_result = benchmark_faiss_ivfpq_mmap(
            base_file, query_vectors, groundtruth, faiss_config, data_dir, faiss_index_file
        )
    
    # Print comparison
    if caliby_result:
        print_comparison_table(caliby_result, faiss_result)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
