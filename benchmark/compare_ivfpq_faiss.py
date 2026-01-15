#!/usr/bin/env python3
"""
Benchmark comparing Caliby IVF+PQ vs FAISS IVF+PQ

Compares:
- Index build time
- Query throughput
- Recall@k
- Memory usage
"""

import numpy as np
import time
import sys
import os
import subprocess
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import tempfile
import shutil

# Add caliby to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import caliby

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: FAISS not installed. Install with: pip install faiss-cpu")


def load_sift1m(base_path='../sift1m'):
    """Load SIFT1M dataset"""
    def read_fvecs(filename):
        with open(filename, 'rb') as f:
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
                vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                yield vec
    
    def read_ivecs(filename):
        with open(filename, 'rb') as f:
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = int(np.frombuffer(dim_bytes, dtype=np.int32)[0])
                vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
                yield vec
    
    base_path = os.path.join(os.path.dirname(__file__), base_path)
    
    print("Loading SIFT1M dataset...")
    
    # Load base vectors (1M)
    base_file = os.path.join(base_path, 'sift_base.fvecs')
    base_vecs = np.array(list(read_fvecs(base_file)))
    print(f"  Base: {base_vecs.shape}")
    
    # Load query vectors
    query_file = os.path.join(base_path, 'sift_query.fvecs')
    query_vecs = np.array(list(read_fvecs(query_file)))
    print(f"  Query: {query_vecs.shape}")
    
    # Load ground truth
    gt_file = os.path.join(base_path, 'sift_groundtruth.ivecs')
    gt = np.array(list(read_ivecs(gt_file)))
    print(f"  Ground truth: {gt.shape}")
    
    return base_vecs, query_vecs, gt


def compute_recall(gt, results, k):
    """Compute recall@k"""
    hits = 0
    for i in range(len(gt)):
        gt_set = set(gt[i][:k])
        result_set = set(results[i][:k])
        hits += len(gt_set & result_set)
    return hits / (len(gt) * k)


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0  # Return 0 if psutil not available


class CalibyIVFPQBenchmark:
    def __init__(self, dim, num_clusters, num_subquantizers):
        self.dim = dim
        self.num_clusters = num_clusters
        self.num_subquantizers = num_subquantizers
        self.index = None
        self.tmpdir = None
        
    def build_index(self, data, train_data=None):
        """Build and train index"""
        if train_data is None:
            train_data = data[:min(100000, len(data))]
        
        self.tmpdir = tempfile.mkdtemp(prefix='caliby_ivfpq_bench_')
        
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        
        caliby.set_buffer_config(32, 32)  # 16GB virtual memory for 1M vectors with 16KB pages
        caliby.open(self.tmpdir, cleanup_if_exist=True)
        
        self.index = caliby.IVFPQIndex(
            max_elements=len(data) + 10000,
            dim=self.dim,
            num_clusters=self.num_clusters,
            num_subquantizers=self.num_subquantizers
        )
        
        # Train
        train_time_start = time.time()
        self.index.train(train_data)
        train_time = time.time() - train_time_start
        
        # Add vectors in batches to avoid issues
        add_time_start = time.time()
        batch_size = 10000
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            self.index.add_points(batch)
            if (i + batch_size) % 50000 == 0:
                print(f"    Added {i+batch_size}/{len(data)} vectors...")
        add_time = time.time() - add_time_start
        
        build_time = time.time() - start_time
        mem_usage = get_memory_usage_mb() - start_mem
        
        return {
            'build_time': build_time,
            'train_time': train_time,
            'add_time': add_time,
            'memory_mb': mem_usage
        }
    
    def search(self, queries, k, nprobe):
        """Search queries"""
        results = []
        
        # Reset stats before search
        self.index.reset_stats()
        
        # Start perf record for profiling
        import subprocess
        perf_output = f"perf_caliby_nprobe{nprobe}.data"
        perf_process = None
        if os.getenv("ENABLE_PERF", "0") == "1":
            print(f"    Starting perf record (output: {perf_output})...")
            perf_process = subprocess.Popen([
                'perf', 'record', '-F', '999', '-g',
                '-o', perf_output,
                '-p', str(os.getpid())
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)  # Give perf time to attach
        
        start_time = time.time()
        for query in queries:
            labels, distances = self.index.search_knn(query, k, nprobe, stats=False)
            results.append(labels)
        elapsed = time.time() - start_time
        
        # Stop perf record
        if perf_process is not None:
            perf_process.terminate()
            perf_process.wait()
            print(f"    Perf recording saved to {perf_output}")
            print(f"    Analyze with: perf report -i {perf_output}")
        
        qps = len(queries) / elapsed
        
        # Get cluster statistics
        stats = self.index.get_stats()
        cluster_sizes = stats['list_sizes']
        vectors_scanned = stats['vectors_scanned']
        
        print(f"  Caliby vectors scanned: {vectors_scanned/len(queries):.0f} avg per query")
        
        return np.array(results), qps, {
            'cluster_sizes': cluster_sizes,
            'vectors_scanned_total': vectors_scanned,
            'vectors_scanned_avg': vectors_scanned / len(queries),
            'min_size': np.min(cluster_sizes) if len(cluster_sizes) > 0 else 0,
            'max_size': np.max(cluster_sizes) if len(cluster_sizes) > 0 else 0,
            'mean_size': np.mean(cluster_sizes) if len(cluster_sizes) > 0 else 0,
            'std_size': np.std(cluster_sizes) if len(cluster_sizes) > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.tmpdir and os.path.exists(self.tmpdir):
            caliby.close()
            shutil.rmtree(self.tmpdir)


class FaissIVFPQBenchmark:
    def __init__(self, dim, num_clusters, num_subquantizers):
        self.dim = dim
        self.num_clusters = num_clusters
        self.num_subquantizers = num_subquantizers
        self.index = None
        
    def build_index(self, data, train_data=None):
        """Build and train index"""
        if not HAS_FAISS:
            return None
            
        if train_data is None:
            train_data = data[:min(100000, len(data))]
        
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        
        # Create IVF+PQ index
        # PQ: product quantizer with M subquantizers and 8 bits per code
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFPQ(
            quantizer,
            self.dim,
            self.num_clusters,
            self.num_subquantizers,
            8  # nbits per subquantizer
        )
        
        # Train
        train_time_start = time.time()
        self.index.train(train_data)
        train_time = time.time() - train_time_start
        
        # Add all vectors
        add_time_start = time.time()
        self.index.add(data)
        add_time = time.time() - add_time_start
        
        build_time = time.time() - start_time
        mem_usage = get_memory_usage_mb() - start_mem
        
        return {
            'build_time': build_time,
            'train_time': train_time,
            'add_time': add_time,
            'memory_mb': mem_usage
        }
    
    def search(self, queries, k, nprobe):
        faiss.omp_set_num_threads(1)
        """Search queries"""
        if not HAS_FAISS:
            return None, 0, {}
            
        self.index.nprobe = nprobe
        
        # Use single-query loop for fair comparison with Caliby
        results = []
        vectors_scanned = 0
        start_time = time.time()
        for query in queries:
            distances, labels = self.index.search(query.reshape(1, -1), k)
            results.append(labels[0])
            # Track vectors scanned (approximation based on nprobe)
            # Each probe scans one inverted list
            for probe_idx in range(nprobe):
                cluster_id = probe_idx  # Simplified - actual clusters depend on query
                vectors_scanned += self.index.invlists.list_size(cluster_id) if probe_idx < self.index.nlist else 0
        elapsed = time.time() - start_time
        
        qps = len(queries) / elapsed
        
        # Get cluster statistics
        cluster_sizes = [self.index.invlists.list_size(i) for i in range(self.index.nlist)]
        
        print(f"  FAISS vectors scanned: {vectors_scanned/len(queries):.0f} avg per query (estimated)")
        
        return np.array(results), qps, {
            'cluster_sizes': cluster_sizes,
            'vectors_scanned_total': vectors_scanned,
            'vectors_scanned_avg': vectors_scanned / len(queries),
            'min_size': np.min(cluster_sizes) if len(cluster_sizes) > 0 else 0,
            'max_size': np.max(cluster_sizes) if len(cluster_sizes) > 0 else 0,
            'mean_size': np.mean(cluster_sizes) if len(cluster_sizes) > 0 else 0,
            'std_size': np.std(cluster_sizes) if len(cluster_sizes) > 0 else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        pass


def run_benchmark(data, queries, ground_truth, k=10, caliby_only=False):
    """Run complete benchmark"""
    dim = data.shape[1]
    
    # Configuration
    num_clusters = 256
    num_subquantizers = 32  # Test generalized SIMD with M=32
    nprobe_values = [1, 8, 16]  # Reduced for faster testing during optimization
    
    print("\n" + "="*80)
    print("BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"Dataset: {len(data)} vectors, {dim} dimensions")
    print(f"Queries: {len(queries)}")
    print(f"Clusters: {num_clusters}")
    print(f"Subquantizers: {num_subquantizers}")
    print(f"k: {k}")
    print(f"nprobe values: {nprobe_values}")
    
    results = {'caliby': {}, 'faiss': {}}
    
    # ==================== CALIBY ====================
    print("\n" + "="*80)
    print("CALIBY IVF+PQ")
    print("="*80)
    
    caliby_bench = CalibyIVFPQBenchmark(dim, num_clusters, num_subquantizers)
    
    print("\n[1/2] Building index...")
    build_stats = caliby_bench.build_index(data)
    results['caliby']['build'] = build_stats
    
    print(f"  Build time: {build_stats['build_time']:.2f}s")
    print(f"    - Train: {build_stats['train_time']:.2f}s")
    print(f"    - Add: {build_stats['add_time']:.2f}s")
    print(f"  Memory: {build_stats['memory_mb']:.1f} MB")
    print(f"  Throughput: {len(data)/build_stats['add_time']:.0f} vectors/s")
    
    print("\n[2/2] Searching...")
    if os.getenv("PERF_SLEEP"):
        import time
        print(f"Sleeping 5s for perf attach (PID={os.getpid()})...")
        time.sleep(5)
    results['caliby']['search'] = {}
    
    for nprobe in nprobe_values:
        search_results, qps, stats = caliby_bench.search(queries, k, nprobe)
        recall = compute_recall(ground_truth, search_results, k)
        
        results['caliby']['search'][nprobe] = {
            'recall': recall,
            'qps': qps,
            'stats': stats
        }
        
        print(f"  nprobe={nprobe:3d}: Recall@{k}={recall:.3f}, QPS={qps:.1f}")
        if nprobe == 16:  # Print stats for nprobe=16
            print(f"    Cluster size: min={stats['min_size']}, max={stats['max_size']}, mean={stats['mean_size']:.0f}, std={stats['std_size']:.0f}")
    
    caliby_bench.cleanup()
        # Skip FAISS if caliby_only flag is set
    if caliby_only:
        print("\nSkipping FAISS benchmark (caliby_only=True)")
        return results
        # ==================== FAISS ====================
    if HAS_FAISS:
        print("\n" + "="*80)
        print("FAISS IVF+PQ")
        print("="*80)
        
        faiss_bench = FaissIVFPQBenchmark(dim, num_clusters, num_subquantizers)
        
        print("\n[1/2] Building index...")
        build_stats = faiss_bench.build_index(data)
        results['faiss']['build'] = build_stats
        
        print(f"  Build time: {build_stats['build_time']:.2f}s")
        print(f"    - Train: {build_stats['train_time']:.2f}s")
        print(f"    - Add: {build_stats['add_time']:.2f}s")
        print(f"  Memory: {build_stats['memory_mb']:.1f} MB")
        print(f"  Throughput: {len(data)/build_stats['add_time']:.0f} vectors/s")
        
        print("\n[2/2] Searching...")
        results['faiss']['search'] = {}
        
        for nprobe in nprobe_values:
            search_results, qps, stats = faiss_bench.search(queries, k, nprobe)
            recall = compute_recall(ground_truth, search_results, k)
            
            results['faiss']['search'][nprobe] = {
                'recall': recall,
                'qps': qps,
                'stats': stats
            }
            
            print(f"  nprobe={nprobe:3d}: Recall@{k}={recall:.3f}, QPS={qps:.1f}")
            if nprobe == 16:  # Print stats for nprobe=16
                print(f"    Cluster size: min={stats['min_size']}, max={stats['max_size']}, mean={stats['mean_size']:.0f}, std={stats['std_size']:.0f}")
        
        faiss_bench.cleanup()
    
    # ==================== COMPARISON ====================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\n--- Build Performance ---")
    caliby_build = results['caliby']['build']['build_time']
    print(f"Caliby: {caliby_build:.2f}s")
    
    if HAS_FAISS:
        faiss_build = results['faiss']['build']['build_time']
        print(f"FAISS:  {faiss_build:.2f}s")
        speedup = faiss_build / caliby_build
        if speedup > 1:
            print(f"→ Caliby is {speedup:.2f}x FASTER")
        else:
            print(f"→ FAISS is {1/speedup:.2f}x faster")
    
    print("\n--- Memory Usage ---")
    caliby_mem = results['caliby']['build']['memory_mb']
    print(f"Caliby: {caliby_mem:.1f} MB")
    
    if HAS_FAISS:
        faiss_mem = results['faiss']['build']['memory_mb']
        print(f"FAISS:  {faiss_mem:.1f} MB")
        ratio = faiss_mem / caliby_mem if caliby_mem > 0 else 0
        if ratio > 1:
            print(f"→ FAISS uses {ratio:.2f}x MORE memory")
        else:
            print(f"→ Caliby uses {1/ratio:.2f}x more memory")
    
   
    
    print("\n--- Recall vs nprobe ---")
    print(f"{'nprobe':<10} {'Caliby':<12} {'FAISS':<12} {'Diff':<10}")
    print("-" * 44)
    for nprobe in nprobe_values:
        caliby_r = results['caliby']['search'][nprobe]['recall']
        line = f"{nprobe:<10} {caliby_r:.3f}        "
        
        if HAS_FAISS:
            faiss_r = results['faiss']['search'][nprobe]['recall']
            diff = caliby_r - faiss_r
            line += f"{faiss_r:.3f}        {diff:+.3f}"
        
        print(line)
    
    return results


def main():
    # Check for dataset
    sift_path = os.path.join(os.path.dirname(__file__), '../sift1m/sift_base.fvecs')
    if not os.path.exists(sift_path):
        print("ERROR: SIFT1M dataset not found!")
        print("Please download it to ../sift1m/")
        print("  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz")
        print("  tar xzf sift.tar.gz")
        return 1
    
    # Load data
    base_vecs, query_vecs, ground_truth = load_sift1m()
    
    # Use fewer queries for faster testing during optimization
    base_vecs = base_vecs  # Full 1M
    query_vecs = query_vecs  # Only 1000 queries for speed
    ground_truth = ground_truth
    
    print(f"\nBenchmarking with: {len(base_vecs)} base vectors, {len(query_vecs)} queries")
    
    # Run benchmark (caliby_only for profiling)
    caliby_only = os.environ.get('CALIBY_ONLY', '0') == '1'
    results = run_benchmark(base_vecs, query_vecs, ground_truth, k=10, caliby_only=caliby_only)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
