#!/usr/bin/env python3
"""
Larger-than-Memory Workload Benchmark: Caliby vs Usearch+mmap

This benchmark compares Caliby's buffer-managed approach against Usearch with mmap
for workloads that exceed available physical memory. Uses SIFT1M dataset.

Test scenarios:
- Limited physical memory (simulated via buffer pool size)
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
import os
import sys
import struct
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import argparse
import psutil

# Add parent directory to path for local caliby build
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing libraries
LIBS_AVAILABLE = {}

try:
    import caliby
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


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    library: str
    memory_limit_mb: int  # Simulated memory limit
    M: int = 16
    ef_construction: int = 100
    ef_search: int = 50
    k: int = 10
    num_threads: int = 1
    enable_prefetch: bool = True
    warmup_queries: int = 100


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    library: str
    config: str
    memory_limit_mb: int
    
    # Build metrics
    build_time: float = 0.0
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


def get_fvecs_dim(filename):
    """Get dimension of vectors in .fvecs file."""
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            raise ValueError(f"Empty file: {filename}")
        return struct.unpack('i', dim_bytes)[0]


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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_file_size_mb(filepath):
    """Get file size in MB."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def is_cgroup_v2():
    """Check if system uses cgroup v2."""
    return os.path.exists('/sys/fs/cgroup/cgroup.controllers')


def setup_cgroup_v1(memory_limit_mb, cgroup_name='hnsw_benchmark'):
    """Setup cgroup v1 memory limit."""
    cgroup_path = f"/sys/fs/cgroup/memory/{cgroup_name}"
    
    try:
        # Create cgroup directory
        subprocess.run(['sudo', 'mkdir', '-p', cgroup_path], check=True)
        
        # Set memory limit
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        subprocess.run(['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {cgroup_path}/memory.limit_in_bytes'], check=True)
        
        # Set swap limit (4x memory)
        memsw_limit_bytes = memory_limit_bytes * 4
        subprocess.run(['sudo', 'sh', '-c', f'echo {memsw_limit_bytes} > {cgroup_path}/memory.memsw.limit_in_bytes'], 
                      check=True, stderr=subprocess.DEVNULL)
        
        print(f"  ✓ Created cgroup v1 at {cgroup_path} with {memory_limit_mb}MB limit")
        return cgroup_path
    except Exception as e:
        print(f"  Warning: Could not setup cgroup: {e}")
        return None


def cleanup_cgroup(cgroup_path):
    """Remove cgroup directory."""
    if cgroup_path and os.path.exists(cgroup_path):
        try:
            subprocess.run(['sudo', 'rmdir', cgroup_path], check=True)
            print(f"  ✓ Cleaned up cgroup: {cgroup_path}")
        except Exception as e:
            print(f"  Warning: Could not cleanup cgroup: {e}")


def drop_caches():
    """Drop OS page caches (requires sudo)."""
    try:
        subprocess.run(['sudo', 'sync'], check=True)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], check=True)
        print("  ✓ Dropped OS page caches")
    except Exception as e:
        print(f"  Warning: Could not drop caches: {e}")


def evict_file_from_cache(filepath):
    """Evict file from OS page cache using vmtouch."""
    try:
        subprocess.run(['vmtouch', '-e', filepath], check=True, capture_output=True)
        print(f"  ✓ Evicted {filepath} from page cache")
    except Exception as e:
        print(f"  Warning: Could not evict file (is vmtouch installed?): {e}")


def run_with_memory_limit(func, memory_limit_mb, *args, **kwargs):
    """Run function under cgroup memory limit (for usearch mmap)."""
    cgroup_name = f"hnsw_benchmark_{os.getpid()}"
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
            print("  ⚠️  Memory limit NOT enforced - OS may use all available page cache")
            print("  ⚠️  This may require sudo privileges or kernel cgroup support")
            return func(*args, **kwargs)
            
    finally:
        # Cleanup: remove process from cgroup (optional, happens automatically on exit)
        # We don't clean up the cgroup directory here to avoid permission issues
        pass


def benchmark_caliby_oom(base_file, query_vectors, groundtruth, config: BenchmarkConfig, 
                          data_dir: str, batch_size: int = 10000) -> BenchmarkResult:
    """
    Benchmark Caliby with constrained buffer pool size using batch-based inserts.
    
    Args:
        base_file: Path to base dataset file (.fvecs)
        query_vectors: Query vectors
        groundtruth: Ground truth nearest neighbors
        config: Benchmark configuration
        data_dir: Directory containing dataset files
        batch_size: Number of vectors to insert per batch (default: 10000)
    """
    print("\n" + "="*80)
    print(f"Benchmarking Caliby (Buffer Limit: {config.memory_limit_mb}MB)")
    print("="*80)
    
    # Get dataset info without loading all vectors
    num_vectors = count_fvecs(base_file)
    dim = get_fvecs_dim(base_file)
    num_queries = query_vectors.shape[0]
    
    print(f"Dataset: {num_vectors} vectors, dim={dim}")
    print(f"Using batch-based inserts: {batch_size} vectors per batch")
    
    result = BenchmarkResult(
        library='caliby',
        config=f'buffer_{config.memory_limit_mb}mb',
        memory_limit_mb=config.memory_limit_mb,
        params={
            'M': config.M,
            'ef_construction': config.ef_construction,
            'ef_search': config.ef_search,
            'num_threads': config.num_threads,
            'enable_prefetch': config.enable_prefetch
        }
    )
    
    # Configure buffer pool size (convert MB to GB for set_buffer_config)
    buffer_size_gb = config.memory_limit_mb / 1024.0
    caliby.set_buffer_config(size_gb=buffer_size_gb)
    print(f"  Buffer pool size: {buffer_size_gb:.2f} GB")
    
    # Build index
    print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
    print(f"  M={config.M}, ef_construction={config.ef_construction}")
    
    rss_before = get_rss_mb()
    start_time = time.time()
    
    index = caliby.HnswIndex(
        max_elements=num_vectors,
        dim=dim,
        M=config.M,
        ef_construction=config.ef_construction,
        enable_prefetch=config.enable_prefetch,
        skip_recovery=False,
        index_id=1,
        name="caliby_oom_benchmark"
    )
    
    # Add vectors in batches (streaming from disk)
    print(f"  Adding {num_vectors} vectors in batches of {batch_size}...")
    total_inserted = 0
    for batch_idx, batch_vectors in enumerate(read_fvecs_batch(base_file, batch_size)):
        index.add_points(batch_vectors)
        total_inserted += len(batch_vectors)
        if batch_idx % 10 == 0:
            print(f"    Progress: {total_inserted}/{num_vectors} ({total_inserted/num_vectors*100:.1f}%)")
    
    print(f"  ✓ Inserted {total_inserted} vectors")
    
    # Flush to disk
    index.flush()
    
    result.build_time = time.time() - start_time
    result.rss_after_build_mb = get_rss_mb()
    
    print(f"✓ Build time: {result.build_time:.2f}s")
    print(f"✓ RSS: {rss_before:.1f} MB -> {result.rss_after_build_mb:.1f} MB")
    
    # Estimate index size from heapfile
    heapfile_path = "heapfile"
    if os.path.exists(heapfile_path):
        result.index_size_mb = get_file_size_mb(heapfile_path)
        print(f"✓ Index size: {result.index_size_mb:.2f} MB")
    
    # Cold cache search - clear buffer and evict heapfile
    print("\n--- Cold Cache Search ---")
    print("Clearing buffer pool and evicting index from OS cache...")
    
    # Delete and recreate index to clear buffer pool
    del index
    time.sleep(1)
    
    # Evict heapfile from OS cache
    if os.path.exists(heapfile_path):
        evict_file_from_cache(heapfile_path)
    
    # Recreate index (will load from disk on-demand)
    index = caliby.HnswIndex(
        max_elements=num_vectors,
        dim=dim,
        M=config.M,
        ef_construction=config.ef_construction,
        enable_prefetch=config.enable_prefetch,
        skip_recovery=False,
        index_id=1,
        name="caliby_oom_benchmark"
    )
    
    print(f"\nCold search (ef_search={config.ef_search})...")
    latencies = []
    all_results = []
    
    start_time = time.time()
    for i in range(num_queries):
        start = time.perf_counter()
        labels, distances = index.search_knn(query_vectors[i], k=config.k, ef_search=config.ef_search)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(labels)
    
    result.cold_qps = num_queries / (time.time() - start_time)
    result.cold_latency_p50 = np.percentile(latencies, 50)
    result.cold_latency_p95 = np.percentile(latencies, 95)
    result.cold_latency_p99 = np.percentile(latencies, 99)
    result.cold_recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
    result.rss_after_cold_mb = get_rss_mb()
    
    print(f"✓ Cold QPS: {result.cold_qps:.2f}")
    print(f"✓ Cold Latency: P50={result.cold_latency_p50:.3f}ms, P95={result.cold_latency_p95:.3f}ms, P99={result.cold_latency_p99:.3f}ms")
    print(f"✓ Cold Recall@{config.k}: {result.cold_recall:.4f}")
    print(f"✓ RSS after cold: {result.rss_after_cold_mb:.1f} MB")
    
    # Warm cache search
    print("\n--- Warm Cache Search ---")
    print(f"Running {config.warmup_queries} warmup queries...")
    for i in range(min(config.warmup_queries, num_queries)):
        _ = index.search_knn(query_vectors[i], k=config.k, ef_search=config.ef_search)
    
    print(f"\nWarm search (ef_search={config.ef_search})...")
    latencies = []
    all_results = []
    
    start_time = time.time()
    for i in range(num_queries):
        start = time.perf_counter()
        labels, distances = index.search_knn(query_vectors[i], k=config.k, ef_search=config.ef_search)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        all_results.append(labels)
    
    result.warm_qps = num_queries / (time.time() - start_time)
    result.warm_latency_p50 = np.percentile(latencies, 50)
    result.warm_latency_p95 = np.percentile(latencies, 95)
    result.warm_latency_p99 = np.percentile(latencies, 99)
    result.warm_recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
    result.rss_after_warm_mb = get_rss_mb()
    
    print(f"✓ Warm QPS: {result.warm_qps:.2f}")
    print(f"✓ Warm Latency: P50={result.warm_latency_p50:.3f}ms, P95={result.warm_latency_p95:.3f}ms, P99={result.warm_latency_p99:.3f}ms")
    print(f"✓ Warm Recall@{config.k}: {result.warm_recall:.4f}")
    print(f"✓ RSS after warm: {result.rss_after_warm_mb:.1f} MB")
    
    return result


def benchmark_usearch_oom(base_file, query_vectors, groundtruth, config: BenchmarkConfig,
                           data_dir: str, index_file: Optional[str] = None, batch_size: int = 10000) -> BenchmarkResult:
    """
    Benchmark Usearch with mmap for larger-than-memory workload using batch-based inserts.
    
    Args:
        base_file: Path to base dataset file (.fvecs)
        query_vectors: Query vectors
        groundtruth: Ground truth nearest neighbors
        config: Benchmark configuration
        data_dir: Directory containing dataset files
        index_file: Path to saved index file (will create if None)
        batch_size: Number of vectors to insert per batch (default: 10000)
    """
    print("\n" + "="*80)
    print(f"Benchmarking Usearch+mmap (Memory Limit: {config.memory_limit_mb}MB)")
    print("="*80)
    
    # Get dataset info without loading all vectors
    num_vectors = count_fvecs(base_file)
    dim = get_fvecs_dim(base_file)
    num_queries = query_vectors.shape[0]
    
    print(f"Dataset: {num_vectors} vectors, dim={dim}")
    print(f"Using batch-based inserts: {batch_size} vectors per batch")
    config.memory_limit_mb += 128
    result = BenchmarkResult(
        library='usearch',
        config=f'mmap_{config.memory_limit_mb}mb',
        memory_limit_mb=config.memory_limit_mb,
        params={
            'M': config.M,
            'ef_construction': config.ef_construction,
            'ef_search': config.ef_search,
            'num_threads': config.num_threads
        }
    )
    
    # Use temporary file if index_file not provided
    if index_file is None:
        index_file = os.path.join(data_dir, 'usearch_oom.index')
    
    # Build or load index
    if not os.path.exists(index_file):
        print(f"\nBuilding index with {num_vectors} vectors (dim={dim})...")
        print(f"  M={config.M}, ef_construction={config.ef_construction}")
        
        # Build with memory limit enforced
        def build_index():
            rss_before = get_rss_mb()
            start_time = time.time()
            
            # Create index
            index = UsearchIndex(
                ndim=dim,
                metric='l2sq',
                dtype='f32',
                connectivity=config.M,
                expansion_add=config.ef_construction,
                expansion_search=config.ef_search
            )
            
            # Add vectors in batches (streaming from disk)
            print(f"  Adding {num_vectors} vectors in batches of {batch_size}...")
            total_inserted = 0
            
            for batch_idx, batch_vectors in enumerate(read_fvecs_batch(base_file, batch_size)):
                batch_labels = np.arange(total_inserted, total_inserted + len(batch_vectors), dtype=np.int64)
                index.add(batch_labels, batch_vectors)
                total_inserted += len(batch_vectors)
                
                if batch_idx % 10 == 0:
                    print(f"    Progress: {total_inserted}/{num_vectors} ({total_inserted/num_vectors*100:.1f}%)")
            
            print(f"  ✓ Inserted {total_inserted} vectors")
            
            build_time = time.time() - start_time
            rss_after = get_rss_mb()
            
            return index, build_time, rss_before, rss_after
        
        # Run build under memory limit
        print(f"\nEnforcing memory limit during index build...")
        index, result.build_time, rss_before, result.rss_after_build_mb = run_with_memory_limit(
            build_index, config.memory_limit_mb
        )
        
        print(f"✓ Build time: {result.build_time:.2f}s")
        print(f"✓ RSS: {rss_before:.1f} MB -> {result.rss_after_build_mb:.1f} MB")
        
        # Save index
        print(f"\nSaving index to {index_file}...")
        save_start = time.time()
        index.save(index_file)
        save_time = time.time() - save_start
        print(f"✓ Saved in {save_time:.2f}s")
        
        del index
        time.sleep(1)
    else:
        print(f"Using existing index: {index_file}")
        result.build_time = 0.0  # Already built
    
    result.index_size_mb = get_file_size_mb(index_file)
    print(f"✓ Index size: {result.index_size_mb:.2f} MB")
    
    # Cold cache search - evict index from OS cache
    print("\n--- Cold Cache Search ---")
    print("Enforcing memory limit for cold cache search...")
    print("Evicting index from OS cache...")
    evict_file_from_cache(index_file)
    
    def cold_search():
        print(f"Loading index with mmap (view=True)...")
        load_start = time.time()
        index = UsearchIndex.restore(index_file, view=True)
        load_time = time.time() - load_start
        rss_after_load = get_rss_mb()
        print(f"✓ Loaded in {load_time:.4f}s (RSS: {rss_after_load:.1f} MB)")
        
        print(f"\nCold search (ef_search={config.ef_search})...")
        latencies = []
        all_results = []
        
        start_time = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            matches = index.search(query_vectors[i], config.k, exact=False)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
            labels = np.array([m.key for m in matches])
            all_results.append(labels)
        
        qps = num_queries / (time.time() - start_time)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
        rss = get_rss_mb()
        
        return index, qps, p50, p95, p99, recall, rss
    
    # Run cold search under memory limit
    index, result.cold_qps, result.cold_latency_p50, result.cold_latency_p95, \
        result.cold_latency_p99, result.cold_recall, result.rss_after_cold_mb = run_with_memory_limit(
            cold_search, config.memory_limit_mb
        )
    
    print(f"✓ Cold QPS: {result.cold_qps:.2f}")
    print(f"✓ Cold Latency: P50={result.cold_latency_p50:.3f}ms, P95={result.cold_latency_p95:.3f}ms, P99={result.cold_latency_p99:.3f}ms")
    print(f"✓ Cold Recall@{config.k}: {result.cold_recall:.4f}")
    print(f"✓ RSS after cold: {result.rss_after_cold_mb:.1f} MB")
    
    # Warm cache search
    print("\n--- Warm Cache Search ---")
    print(f"Running {config.warmup_queries} warmup queries...")
    for i in range(min(config.warmup_queries, num_queries)):
        _ = index.search(query_vectors[i], config.k, exact=False)
    
    print(f"\nWarm search (ef_search={config.ef_search})...")
    latencies = []
    all_results = []
    
    start_time = time.time()
    for i in range(num_queries):
        start = time.perf_counter()
        matches = index.search(query_vectors[i], config.k, exact=False)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)
        labels = np.array([m.key for m in matches])
        all_results.append(labels)
    
    result.warm_qps = num_queries / (time.time() - start_time)
    result.warm_latency_p50 = np.percentile(latencies, 50)
    result.warm_latency_p95 = np.percentile(latencies, 95)
    result.warm_latency_p99 = np.percentile(latencies, 99)
    result.warm_recall = compute_recall_at_k(np.array(all_results), groundtruth, k=config.k)
    result.rss_after_warm_mb = get_rss_mb()
    
    print(f"✓ Warm QPS: {result.warm_qps:.2f}")
    print(f"✓ Warm Latency: P50={result.warm_latency_p50:.3f}ms, P95={result.warm_latency_p95:.3f}ms, P99={result.warm_latency_p99:.3f}ms")
    print(f"✓ Warm Recall@{config.k}: {result.warm_recall:.4f}")
    print(f"✓ RSS after warm: {result.rss_after_warm_mb:.1f} MB")
    
    return result


def print_comparison_table(results: List[BenchmarkResult]):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Library':<15} {'Config':<20} {'Mem(MB)':<10} {'Build(s)':<10} {'Size(MB)':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.library:<15} {r.config:<20} {r.memory_limit_mb:<10} {r.build_time:<10.2f} {r.index_size_mb:<10.2f}")
    
    # Cold cache metrics
    print(f"\n{'COLD CACHE PERFORMANCE'}")
    print("-" * 100)
    print(f"{'Library':<15} {'QPS':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Recall':<10} {'RSS(MB)':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.library:<15} {r.cold_qps:<10.2f} {r.cold_latency_p50:<10.3f} {r.cold_latency_p95:<10.3f} "
              f"{r.cold_latency_p99:<10.3f} {r.cold_recall:<10.4f} {r.rss_after_cold_mb:<10.1f}")
    
    # Warm cache metrics
    print(f"\n{'WARM CACHE PERFORMANCE'}")
    print("-" * 100)
    print(f"{'Library':<15} {'QPS':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Recall':<10} {'RSS(MB)':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.library:<15} {r.warm_qps:<10.2f} {r.warm_latency_p50:<10.3f} {r.warm_latency_p95:<10.3f} "
              f"{r.warm_latency_p99:<10.3f} {r.warm_recall:<10.4f} {r.rss_after_warm_mb:<10.1f}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark larger-than-memory workload: Caliby vs Usearch+mmap'
    )
    parser.add_argument('--data-dir', type=str, default='./sift1m',
                        help='Directory containing SIFT1M dataset files')
    parser.add_argument('--memory-limits', type=int, nargs='+', default=[256],
                        help='Memory limits to test (MB)')
    parser.add_argument('--M', type=int, default=16,
                        help='HNSW M parameter')
    parser.add_argument('--ef-construction', type=int, default=100,
                        help='HNSW ef_construction parameter')
    parser.add_argument('--ef-search', type=int, default=50,
                        help='HNSW ef_search parameter')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of nearest neighbors')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads for search')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Number of warmup queries')
    parser.add_argument('--libraries', type=str, nargs='+', default=['caliby', 'usearch'],
                        help='Libraries to benchmark (caliby, usearch)')
    parser.add_argument('--usearch-index', type=str, default=None,
                        help='Path to pre-built usearch index file')
    
    args = parser.parse_args()
    
    # Check library availability
    for lib in args.libraries:
        if not LIBS_AVAILABLE.get(lib, False):
            print(f"Error: {lib} not available")
            sys.exit(1)
    
    # Load SIFT1M dataset
    print("="*80)
    print("LARGER-THAN-MEMORY BENCHMARK: Caliby vs Usearch+mmap")
    print("="*80)
    print(f"\nDataset: SIFT1M")
    print(f"Memory limits: {args.memory_limits} MB")
    print(f"Parameters: M={args.M}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    print(f"Libraries: {', '.join(args.libraries)}")
    
    data_dir = Path(args.data_dir)
    base_file = data_dir / 'sift_base.fvecs'
    query_file = data_dir / 'sift_query.fvecs'
    gt_file = data_dir / 'sift_groundtruth.ivecs'
    
    if not (base_file.exists() and query_file.exists() and gt_file.exists()):
        print(f"\nError: SIFT1M dataset not found in {data_dir}")
        print("Expected files: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs")
        print("\nDownload from: ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz")
        sys.exit(1)
    
    print(f"\nLoading SIFT1M query vectors and ground truth from {data_dir}...")
    print("  (Base vectors will be streamed in batches during index building)")
    query_vectors = read_fvecs(str(query_file))
    groundtruth = read_ivecs(str(gt_file))
    
    # Get base dataset info without loading into memory
    num_base = count_fvecs(str(base_file))
    base_dim = get_fvecs_dim(str(base_file))
    
    print(f"  Base vectors: {num_base} x {base_dim} (streaming)")
    print(f"  Query vectors: {query_vectors.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    # Run benchmarks
    all_results = []
    
    for memory_limit_mb in args.memory_limits:
        print(f"\n{'#'*80}")
        print(f"# Testing with {memory_limit_mb} MB memory limit")
        print(f"{'#'*80}")
        
        config = BenchmarkConfig(
            library='',  # Will be set per benchmark
            memory_limit_mb=memory_limit_mb,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            k=args.k,
            num_threads=args.threads,
            warmup_queries=args.warmup
        )
        
        for lib in args.libraries:
            config.library = lib
            
            try:
                if lib == 'caliby':
                    result = benchmark_caliby_oom(str(base_file), query_vectors, groundtruth, 
                                                   config, str(data_dir), batch_size=10000)
                    all_results.append(result)
                    
                elif lib == 'usearch':
                    result = benchmark_usearch_oom(str(base_file), query_vectors, groundtruth,
                                                    config, str(data_dir), args.usearch_index, batch_size=10000)
                    all_results.append(result)
                
                # Cleanup between runs
                time.sleep(2)
                
            except Exception as e:
                print(f"\n❌ Error benchmarking {lib}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    if all_results:
        print_comparison_table(all_results)
        
        # Save results to file
        output_file = data_dir / 'oom_benchmark_results.txt'
        with open(output_file, 'w') as f:
            f.write("LARGER-THAN-MEMORY BENCHMARK RESULTS\n")
            f.write("="*100 + "\n\n")
            for r in all_results:
                f.write(f"{r.library} ({r.config}):\n")
                f.write(f"  Build: {r.build_time:.2f}s, Size: {r.index_size_mb:.2f}MB\n")
                f.write(f"  Cold: QPS={r.cold_qps:.2f}, P50={r.cold_latency_p50:.3f}ms, Recall={r.cold_recall:.4f}\n")
                f.write(f"  Warm: QPS={r.warm_qps:.2f}, P50={r.warm_latency_p50:.3f}ms, Recall={r.warm_recall:.4f}\n")
                f.write("\n")
        
        print(f"\n✓ Results saved to: {output_file}")
    else:
        print("\n❌ No results generated")


if __name__ == '__main__':
    main()
