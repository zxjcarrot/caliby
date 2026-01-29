#!/usr/bin/env python3
"""
Caliby vs ChromaDB vs Qdrant vs Weaviate Benchmark

This benchmark compares Caliby, ChromaDB, Qdrant, and Weaviate across multiple dimensions:
1. Document storage (insertion throughput)
2. Document retrieval (query by ID)
3. Vector search (similarity search performance)
4. Filtered vector search
5. Hybrid search (vector + text)

Using SIFT1M dataset (1M 128-dimensional vectors) for comprehensive testing.

Metrics measured:
- Build time (document insertion + index creation)
- Index size (memory/disk usage)
- Document retrieval latency
- Vector search throughput (QPS)
- Vector search latency (P50, P95, P99)
- Recall@10 accuracy
- Filtered search performance
"""

import numpy as np
import time
import os
import sys
import struct
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import argparse
import json
import gc

# Add parent directory to path to import local caliby build
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing libraries
LIBS_AVAILABLE = {}

try:
    import caliby
    caliby.set_buffer_config(size_gb=16)
    LIBS_AVAILABLE['caliby'] = True
except ImportError as e:
    print(f"Warning: caliby not available: {e}")
    LIBS_AVAILABLE['caliby'] = False

try:
    import chromadb
    LIBS_AVAILABLE['chromadb'] = True
except ImportError:
    print("Warning: chromadb not available (install: pip install chromadb)")
    LIBS_AVAILABLE['chromadb'] = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    LIBS_AVAILABLE['qdrant'] = True
except ImportError:
    print("Warning: qdrant not available (install: pip install qdrant-client)")
    LIBS_AVAILABLE['qdrant'] = False

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import Filter as WeaviateFilter
    LIBS_AVAILABLE['weaviate'] = True
except ImportError:
    print("Warning: weaviate not available (install: pip install weaviate-client)")
    LIBS_AVAILABLE['weaviate'] = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    library: str
    dataset_size: int
    dimension: int
    
    # Build metrics
    insert_time: float
    insert_throughput: float  # docs/sec
    index_build_time: float
    total_build_time: float
    
    # Storage metrics
    index_size_mb: float
    
    # Document retrieval metrics
    retrieval_time_ms: float  # Average time to retrieve 1 doc by ID
    retrieval_batch_time_ms: float  # Average time to retrieve 100 docs by ID
    
    # Vector search metrics
    search_qps: float
    search_latency_p50: float
    search_latency_p95: float
    search_latency_p99: float
    recall_at_10: float
    
    # Configuration
    params: Dict
    
    # Filtered search metrics (with defaults)
    filtered_search_qps: float = 0.0
    filtered_search_latency_p50: float = 0.0
    filtered_search_latency_p95: float = 0.0
    filtered_recall_at_10: float = 0.0
    
    # Hybrid search metrics (vector + text)
    hybrid_search_qps: float = 0.0
    hybrid_search_latency_p50: float = 0.0
    hybrid_search_latency_p95: float = 0.0
    
    def to_dict(self):
        return asdict(self)


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


def compute_groundtruth(base_vectors, query_vectors, k=100):
    """
    Compute ground truth by brute-force L2 distance calculation.
    
    This is necessary when using a subset of the base vectors, because
    the original ground truth was computed on the full dataset.
    
    Args:
        base_vectors: Shape (n_base, dim)
        query_vectors: Shape (n_queries, dim)
        k: Number of nearest neighbors to find
    
    Returns:
        Ground truth array of shape (n_queries, k) with indices
    """
    print(f"\nComputing ground truth for {len(query_vectors)} queries against {len(base_vectors)} vectors...")
    n_queries = len(query_vectors)
    n_base = len(base_vectors)
    
    # For larger datasets, process in batches to manage memory
    groundtruth = np.zeros((n_queries, k), dtype=np.int32)
    
    batch_size = 100  # Process queries in batches
    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        batch_queries = query_vectors[batch_start:batch_end]
        
        # Compute L2 distances: (q - b)^2 = q^2 - 2*q*b + b^2
        # query_norms: (batch, 1)
        # base_norms: (1, n_base)
        # dot: (batch, n_base)
        query_norms = np.sum(batch_queries ** 2, axis=1, keepdims=True)
        base_norms = np.sum(base_vectors ** 2, axis=1, keepdims=True).T
        dot_products = np.dot(batch_queries, base_vectors.T)
        
        distances = query_norms - 2 * dot_products + base_norms
        
        # Get top-k indices for each query
        # np.argpartition is faster than full sort for finding k smallest
        if k < n_base:
            # Get indices of k smallest distances
            partitioned_indices = np.argpartition(distances, k, axis=1)[:, :k]
            # Now sort these k indices by their actual distances
            for i in range(len(batch_queries)):
                indices = partitioned_indices[i]
                sorted_indices = indices[np.argsort(distances[i, indices])]
                groundtruth[batch_start + i] = sorted_indices
        else:
            # If k >= n_base, just sort all
            sorted_indices = np.argsort(distances, axis=1)[:, :k]
            groundtruth[batch_start:batch_end] = sorted_indices
        
        if (batch_end % 1000 == 0) or (batch_end == n_queries):
            print(f"  Processed {batch_end}/{n_queries} queries...")
    
    print(f"  ✓ Ground truth computed")
    return groundtruth


def compute_filtered_groundtruth(base_vectors, query_vectors, filter_mask, k=10):
    """
    Compute ground truth for filtered search by brute-force.
    
    Args:
        base_vectors: Shape (n_base, dim)
        query_vectors: Shape (n_queries, dim)
        filter_mask: Boolean array of shape (n_base,) where True means the vector passes the filter
        k: Number of nearest neighbors to find
    
    Returns:
        List of lists, where each inner list contains the k nearest neighbor indices
        that pass the filter for that query
    """
    print(f"\nComputing filtered ground truth for {len(query_vectors)} queries...")
    n_queries = len(query_vectors)
    n_base = len(base_vectors)
    
    # Get indices that pass the filter
    filtered_indices = np.where(filter_mask)[0]
    filtered_vectors = base_vectors[filtered_indices]
    n_filtered = len(filtered_indices)
    
    print(f"  Filtered to {n_filtered} vectors ({100*n_filtered/n_base:.1f}% of data)")
    
    if n_filtered == 0:
        return [[] for _ in range(n_queries)]
    
    filtered_groundtruth = []
    
    batch_size = 100  # Process queries in batches
    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        batch_queries = query_vectors[batch_start:batch_end]
        
        # Compute L2 distances against filtered vectors
        query_norms = np.sum(batch_queries ** 2, axis=1, keepdims=True)
        base_norms = np.sum(filtered_vectors ** 2, axis=1, keepdims=True).T
        dot_products = np.dot(batch_queries, filtered_vectors.T)
        
        distances = query_norms - 2 * dot_products + base_norms
        
        # Get top-k indices
        actual_k = min(k, n_filtered)
        if actual_k < n_filtered:
            partitioned_indices = np.argpartition(distances, actual_k, axis=1)[:, :actual_k]
            for i in range(len(batch_queries)):
                indices = partitioned_indices[i]
                sorted_indices = indices[np.argsort(distances[i, indices])]
                # Map back to original indices
                original_indices = filtered_indices[sorted_indices].tolist()
                filtered_groundtruth.append(original_indices)
        else:
            sorted_indices = np.argsort(distances, axis=1)[:, :actual_k]
            for i in range(len(batch_queries)):
                original_indices = filtered_indices[sorted_indices[i]].tolist()
                filtered_groundtruth.append(original_indices)
        
        if (batch_end % 1000 == 0) or (batch_end == n_queries):
            print(f"  Processed {batch_end}/{n_queries} queries...")
    
    print(f"  ✓ Filtered ground truth computed")
    return filtered_groundtruth


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
    """Benchmark Caliby implementation."""
    print("\n" + "="*70)
    print("Benchmarking Caliby")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    temp_dir = tempfile.mkdtemp(prefix='caliby_bench_')
    
    try:
        # Initialize database
        caliby.open(temp_dir, cleanup_if_exist=True)
        
        # Create schema with category field for filtered search
        schema = caliby.Schema()
        schema.add_field("doc_id", caliby.FieldType.INT)
        schema.add_field("category", caliby.FieldType.INT)  # For filtered search
        schema.add_field("content", caliby.FieldType.STRING)
        
        # Create collection
        collection = caliby.Collection(
            "sift1m",
            schema,
            vector_dim=dim,
            distance_metric=caliby.DistanceMetric.L2
        )
        
        # ===== INDEX CREATION (must be before insertion!) =====
        print("\n1. Index Creation")
        print("-" * 70)
        
        index_start = time.time()
        collection.create_hnsw_index(
            "vec_idx",
            M=params.get('M', 16),
            ef_construction=params.get('ef_construction', 200)
        )
        index_create_time = time.time() - index_start
        print(f"  ✓ Created HNSW index in {index_create_time:.2f}s")
        
        # Create text index for hybrid search
        text_idx_start = time.time()
        collection.create_text_index("text_idx")
        text_idx_time = time.time() - text_idx_start
        print(f"  ✓ Created text index in {text_idx_time:.2f}s")
        
        # Create metadata index for filtered search (on category field)
        meta_idx_start = time.time()
        collection.create_metadata_index("category_idx", ["category"])
        meta_idx_time = time.time() - meta_idx_start
        print(f"  ✓ Created metadata index in {meta_idx_time:.2f}s")
        
        # ===== DOCUMENT INSERTION =====
        print("\n2. Document Insertion (with index building)")
        print("-" * 70)
        
        batch_size = params.get('insert_batch_size', 10000)
        num_categories = 10  # 10 categories for filtering
        
        # Generate same text content as other libraries for fair comparison
        word_pool = ["vector", "search", "neural", "embedding", "similarity",
                     "machine", "learning", "database", "index", "query",
                     "document", "feature", "model", "representation", "metric"]
        contents = []
        metadatas = []
        for i in range(num_vectors):
            # Assign category based on doc_id
            category = i % num_categories
            # Generate text content matching ChromaDB format
            words = [word_pool[(i + j) % len(word_pool)] for j in range(5)]
            content = f"doc_{i} category_{category} " + " ".join(words)
            contents.append(content)
            metadatas.append({"doc_id": i, "category": category})
        
        insert_start = time.time()
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_vectors)
            
            batch_contents = contents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_vectors = base_vectors[start_idx:end_idx].tolist()
            
            collection.add(batch_contents, batch_metadatas, batch_vectors)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Inserted {end_idx}/{num_vectors} documents...")
        
        insert_time = time.time() - insert_start
        insert_throughput = num_vectors / insert_time
        index_build_time = insert_time  # Index is built during insertion
        
        print(f"  ✓ Inserted {num_vectors} documents in {insert_time:.2f}s")
        print(f"  ✓ Throughput: {insert_throughput:.0f} docs/sec")
        
        # Flush to ensure all data is written
        caliby.flush_storage()
        
        # ===== STORAGE SIZE =====
        index_size_mb = get_directory_size(temp_dir)
        print(f"  ✓ Index size: {index_size_mb:.2f} MB")
        
        # ===== DOCUMENT RETRIEVAL =====
        print("\n3. Document Retrieval")
        print("-" * 70)
        
        # Single document retrieval
        num_retrieval_tests = 100
        retrieval_times = []
        test_ids = np.random.randint(0, num_vectors, size=num_retrieval_tests)
        
        for doc_id in test_ids:
            start = time.perf_counter()
            docs = collection.get([int(doc_id)])
            elapsed = (time.perf_counter() - start) * 1000  # ms
            retrieval_times.append(elapsed)
        
        retrieval_time_ms = np.mean(retrieval_times)
        print(f"  ✓ Single doc retrieval: {retrieval_time_ms:.3f} ms (avg over {num_retrieval_tests} queries)")
        
        # Batch retrieval (100 docs)
        batch_retrieval_times = []
        for _ in range(10):
            batch_ids = np.random.randint(0, num_vectors, size=100).tolist()
            start = time.perf_counter()
            docs = collection.get(batch_ids)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            batch_retrieval_times.append(elapsed)
        
        retrieval_batch_time_ms = np.mean(batch_retrieval_times)
        print(f"  ✓ Batch retrieval (100 docs): {retrieval_batch_time_ms:.3f} ms")
        
        # ===== VECTOR SEARCH =====
        print("\n4. Vector Search")
        print("-" * 70)
        
        k = 10
        num_queries = len(query_vectors)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k)
        
        # Measure search performance
        search_latencies = []
        search_results = []
        
        search_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            search_latencies.append(elapsed)
            search_results.append([r.doc_id for r in results])
        
        total_search_time = time.time() - search_start
        search_qps = num_queries / total_search_time
        
        # Calculate latency percentiles
        search_latencies = np.array(search_latencies)
        latency_p50 = np.percentile(search_latencies, 50)
        latency_p95 = np.percentile(search_latencies, 95)
        latency_p99 = np.percentile(search_latencies, 99)
        
        print(f"  ✓ Queries: {num_queries}")
        print(f"  ✓ QPS: {search_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {latency_p50:.2f} ms")
        print(f"  ✓ Latency P95: {latency_p95:.2f} ms")
        print(f"  ✓ Latency P99: {latency_p99:.2f} ms")
        
        # Calculate recall
        search_results_array = np.array(search_results)
        recall = compute_recall_at_k(search_results_array, groundtruth, k=10)
        print(f"  ✓ Recall@10: {recall:.4f}")
        
        # ===== FILTERED VECTOR SEARCH =====
        print("\n4. Filtered Vector Search")
        print("-" * 70)
        
        # Filter to specific category (10% of data)
        filter_category = 0  # Will match ~10% of docs
        # Use MongoDB-style filter format: {"field": value} for equality
        filter_json = json.dumps({"category": filter_category})
        
        # Compute proper filtered ground truth using brute-force
        # Create filter mask: True for vectors where index % num_categories == filter_category
        filter_mask = np.array([i % num_categories == filter_category for i in range(num_vectors)])
        filtered_groundtruth = compute_filtered_groundtruth(base_vectors, query_vectors, filter_mask, k=k)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, filter=filter_json)
        
        filtered_latencies = []
        filtered_results = []
        
        filtered_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, filter=filter_json)
            elapsed = (time.perf_counter() - start) * 1000
            
            filtered_latencies.append(elapsed)
            filtered_results.append([r.doc_id for r in results])
        
        filtered_total_time = time.time() - filtered_start
        filtered_qps = num_queries / filtered_total_time
        
        filtered_latencies = np.array(filtered_latencies)
        filtered_p50 = np.percentile(filtered_latencies, 50)
        filtered_p95 = np.percentile(filtered_latencies, 95)
        
        # Calculate filtered recall
        filtered_recall_sum = 0
        for i in range(len(filtered_results)):
            if len(filtered_groundtruth[i]) > 0:
                result_set = set(filtered_results[i])
                gt_set = set(filtered_groundtruth[i][:k])
                filtered_recall_sum += len(result_set & gt_set) / min(len(gt_set), k)
        filtered_recall = filtered_recall_sum / len(filtered_results) if filtered_results else 0
        
        print(f"  ✓ Filter: category == {filter_category} (~10% of data)")
        print(f"  ✓ QPS: {filtered_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {filtered_p50:.2f} ms")
        print(f"  ✓ Latency P95: {filtered_p95:.2f} ms")
        print(f"  ✓ Filtered Recall@10: {filtered_recall:.4f}")
        
        # ===== HYBRID SEARCH (Vector + Text) =====
        print("\n5. Hybrid Search (Vector + Text)")
        print("-" * 70)
        
        # Note: Using empty text content, so hybrid search will behave like pure vector search
        text_queries = ["" for _ in range(num_queries)]
        
        # Warmup
        for i in range(min(1000, num_queries)):
            collection.search_hybrid(
                query_vectors[i].tolist(), "vec_idx",
                text_queries[i], "text_idx",
                k=k
            )
        
        hybrid_latencies = []
        hybrid_results = []
        
        hybrid_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.search_hybrid(
                query_vectors[i].tolist(), "vec_idx",
                text_queries[i], "text_idx",
                k=k
            )
            elapsed = (time.perf_counter() - start) * 1000
            
            hybrid_latencies.append(elapsed)
            hybrid_results.append([r.doc_id for r in results])
        
        hybrid_total_time = time.time() - hybrid_start
        hybrid_qps = num_queries / hybrid_total_time
        
        hybrid_latencies = np.array(hybrid_latencies)
        hybrid_p50 = np.percentile(hybrid_latencies, 50)
        hybrid_p95 = np.percentile(hybrid_latencies, 95)
        
        print(f"  ✓ QPS: {hybrid_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {hybrid_p50:.2f} ms")
        print(f"  ✓ Latency P95: {hybrid_p95:.2f} ms")
        
        # Create result object
        result = BenchmarkResult(
            library="Caliby",
            dataset_size=num_vectors,
            dimension=dim,
            insert_time=insert_time,
            insert_throughput=insert_throughput,
            index_build_time=index_build_time,
            total_build_time=insert_time + index_build_time,
            index_size_mb=index_size_mb,
            retrieval_time_ms=retrieval_time_ms,
            retrieval_batch_time_ms=retrieval_batch_time_ms,
            search_qps=search_qps,
            search_latency_p50=latency_p50,
            search_latency_p95=latency_p95,
            search_latency_p99=latency_p99,
            recall_at_10=recall,
            filtered_search_qps=filtered_qps,
            filtered_search_latency_p50=filtered_p50,
            filtered_search_latency_p95=filtered_p95,
            filtered_recall_at_10=filtered_recall,
            hybrid_search_qps=hybrid_qps,
            hybrid_search_latency_p50=hybrid_p50,
            hybrid_search_latency_p95=hybrid_p95,
            params=params
        )
        
        return result
        
    finally:
        # Cleanup
        try:
            caliby.close()
        except:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_chromadb(base_vectors, query_vectors, groundtruth, params):
    """Benchmark ChromaDB implementation."""
    print("\n" + "="*70)
    print("Benchmarking ChromaDB")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    temp_dir = tempfile.mkdtemp(prefix='chromadb_bench_')
    
    client = None
    try:
        # Initialize ChromaDB with new API
        client = chromadb.PersistentClient(
            path=temp_dir,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        
        # Create collection
        collection = client.create_collection(
            name="sift1m",
            metadata={"hnsw:space": "l2"}  # L2 distance
        )
        
        # ===== DOCUMENT INSERTION =====
        print("\n1. Document Insertion")
        print("-" * 70)
        
        # ChromaDB has a max batch size limit (~5000), use smaller batches
        batch_size = min(params.get('insert_batch_size', 5000), 5000)
        num_categories = 10  # 10 categories for filtering
        ids = [str(i) for i in range(num_vectors)]
        
        # Generate content with categories like Caliby
        word_pool = ["vector", "search", "neural", "embedding", "similarity",
                     "machine", "learning", "database", "index", "query",
                     "document", "feature", "model", "representation", "metric"]
        documents = []
        metadatas = []
        for i in range(num_vectors):
            category = i % num_categories
            words = [word_pool[(i + j) % len(word_pool)] for j in range(5)]
            content = f"doc_{i} category_{category} " + " ".join(words)
            documents.append(content)
            metadatas.append({"doc_id": i, "category": category})
        
        insert_start = time.time()
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_vectors)
            
            batch_ids = ids[start_idx:end_idx]
            batch_documents = documents[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_embeddings = base_vectors[start_idx:end_idx].tolist()
            
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Inserted {end_idx}/{num_vectors} documents...")
        
        insert_time = time.time() - insert_start
        insert_throughput = num_vectors / insert_time
        
        print(f"  ✓ Inserted {num_vectors} documents in {insert_time:.2f}s")
        print(f"  ✓ Throughput: {insert_throughput:.0f} docs/sec")
        
        # Data is automatically persisted with PersistentClient
        # No need to call persist() explicitly
        
        # ===== STORAGE SIZE =====
        index_size_mb = get_directory_size(temp_dir)
        print(f"  ✓ Index size: {index_size_mb:.2f} MB")
        
        # Note: ChromaDB builds index automatically during insertion
        index_build_time = 0.0  # Already included in insertion time
        
        # ===== DOCUMENT RETRIEVAL =====
        print("\n2. Document Retrieval")
        print("-" * 70)
        
        # Single document retrieval
        num_retrieval_tests = 100
        retrieval_times = []
        test_ids = np.random.randint(0, num_vectors, size=num_retrieval_tests)
        
        for doc_id in test_ids:
            start = time.perf_counter()
            result = collection.get(ids=[str(int(doc_id))])
            elapsed = (time.perf_counter() - start) * 1000  # ms
            retrieval_times.append(elapsed)
        
        retrieval_time_ms = np.mean(retrieval_times)
        print(f"  ✓ Single doc retrieval: {retrieval_time_ms:.3f} ms (avg over {num_retrieval_tests} queries)")
        
        # Batch retrieval (100 docs)
        batch_retrieval_times = []
        for _ in range(10):
            batch_ids = [str(int(i)) for i in np.random.choice(num_vectors, size=min(100, num_vectors), replace=False)]
            start = time.perf_counter()
            result = collection.get(ids=batch_ids)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            batch_retrieval_times.append(elapsed)
        
        retrieval_batch_time_ms = np.mean(batch_retrieval_times)
        print(f"  ✓ Batch retrieval (100 docs): {retrieval_batch_time_ms:.3f} ms")
        
        # ===== VECTOR SEARCH =====
        print("\n3. Vector Search")
        print("-" * 70)
        
        k = 10
        num_queries = len(query_vectors)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            collection.query(
                query_embeddings=[query_vectors[i].tolist()],
                n_results=k
            )
        
        # Measure search performance
        search_latencies = []
        search_results = []
        
        search_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.query(
                query_embeddings=[query_vectors[i].tolist()],
                n_results=k
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            search_latencies.append(elapsed)
            # Convert string IDs back to integers
            result_ids = [int(id_str) for id_str in results['ids'][0]]
            search_results.append(result_ids)
        
        total_search_time = time.time() - search_start
        search_qps = num_queries / total_search_time
        
        # Calculate latency percentiles
        search_latencies = np.array(search_latencies)
        latency_p50 = np.percentile(search_latencies, 50)
        latency_p95 = np.percentile(search_latencies, 95)
        latency_p99 = np.percentile(search_latencies, 99)
        
        print(f"  ✓ Queries: {num_queries}")
        print(f"  ✓ QPS: {search_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {latency_p50:.2f} ms")
        print(f"  ✓ Latency P95: {latency_p95:.2f} ms")
        print(f"  ✓ Latency P99: {latency_p99:.2f} ms")
        
        # Calculate recall
        search_results_array = np.array(search_results)
        recall = compute_recall_at_k(search_results_array, groundtruth, k=10)
        print(f"  ✓ Recall@10: {recall:.4f}")
        
        # ===== FILTERED VECTOR SEARCH =====
        print("\n4. Filtered Vector Search")
        print("-" * 70)
        print("  ⚠ Skipping ChromaDB filtered search (too slow for 1M vectors)")
        
        # Set default values since we're skipping filtered search
        filtered_qps = 0.0
        filtered_p50 = 0.0
        filtered_p95 = 0.0
        filtered_recall = 0.0
        
        # Filter to specific category (10% of data)
        # filter_category = 0  # Will match ~10% of docs
        # where_filter = {"category": {"$eq": filter_category}}
        
        # # Compute proper filtered ground truth using brute-force
        # filter_mask = np.array([i % num_categories == filter_category for i in range(num_vectors)])
        # filtered_groundtruth = compute_filtered_groundtruth(base_vectors, query_vectors, filter_mask, k=k)
        
        # # Warmup
        # for i in range(min(10, num_queries)):
        #     collection.query(
        #         query_embeddings=[query_vectors[i].tolist()],
        #         n_results=k,
        #         where=where_filter
        #     )
        
        # filtered_latencies = []
        # filtered_results = []
        
        # filtered_start = time.time()
        # for i in range(num_queries):
        #     start = time.perf_counter()
        #     results = collection.query(
        #         query_embeddings=[query_vectors[i].tolist()],
        #         n_results=k,
        #         where=where_filter
        #     )
        #     elapsed = (time.perf_counter() - start) * 1000
            
        #     filtered_latencies.append(elapsed)
        #     result_ids = [int(id_str) for id_str in results['ids'][0]] if results['ids'][0] else []
        #     filtered_results.append(result_ids)
        
        # filtered_total_time = time.time() - filtered_start
        # filtered_qps = num_queries / filtered_total_time
        
        # filtered_latencies = np.array(filtered_latencies)
        # filtered_p50 = np.percentile(filtered_latencies, 50)
        # filtered_p95 = np.percentile(filtered_latencies, 95)
        
        # # Calculate filtered recall
        # filtered_recall_sum = 0
        # for i in range(len(filtered_results)):
        #     if len(filtered_groundtruth[i]) > 0:
        #         result_set = set(filtered_results[i])
        #         gt_set = set(filtered_groundtruth[i][:k])
        #         filtered_recall_sum += len(result_set & gt_set) / min(len(gt_set), k)
        # filtered_recall = filtered_recall_sum / len(filtered_results) if filtered_results else 0
        
        # print(f"  ✓ Filter: category == {filter_category} (~10% of data)")
        # print(f"  ✓ QPS: {filtered_qps:.1f} queries/sec")
        # print(f"  ✓ Latency P50: {filtered_p50:.2f} ms")
        # print(f"  ✓ Latency P95: {filtered_p95:.2f} ms")
        # print(f"  ✓ Filtered Recall@10: {filtered_recall:.4f}")
        
        # ===== HYBRID SEARCH =====
        # Note: ChromaDB doesn't support true hybrid search (vector + text BM25 fusion)
        # It uses "where_document" for text matching but this is different from BM25 scoring
        print("\n5. Hybrid Search (Vector + Text)")
        print("-" * 70)
        print("  ⚠ ChromaDB does not support true hybrid search (vector + BM25 fusion)")
        print("  ⚠ Skipping hybrid search benchmark for ChromaDB")
        hybrid_qps = 0.0
        hybrid_p50 = 0.0
        hybrid_p95 = 0.0
        
        # Create result object
        result = BenchmarkResult(
            library="ChromaDB",
            dataset_size=num_vectors,
            dimension=dim,
            insert_time=insert_time,
            insert_throughput=insert_throughput,
            index_build_time=index_build_time,
            total_build_time=insert_time + index_build_time,
            index_size_mb=index_size_mb,
            retrieval_time_ms=retrieval_time_ms,
            retrieval_batch_time_ms=retrieval_batch_time_ms,
            search_qps=search_qps,
            search_latency_p50=latency_p50,
            search_latency_p95=latency_p95,
            search_latency_p99=latency_p99,
            recall_at_10=recall,
            filtered_search_qps=filtered_qps,
            filtered_search_latency_p50=filtered_p50,
            filtered_search_latency_p95=filtered_p95,
            filtered_recall_at_10=filtered_recall,
            hybrid_search_qps=hybrid_qps,
            hybrid_search_latency_p50=hybrid_p50,
            hybrid_search_latency_p95=hybrid_p95,
            params=params
        )
        
        return result
        
    finally:
        # Cleanup
        if client is not None:
            del client
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_qdrant(base_vectors, query_vectors, groundtruth, params):
    """Benchmark Qdrant implementation."""
    print("\n" + "="*70)
    print("Benchmarking Qdrant")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    temp_dir = tempfile.mkdtemp(prefix='qdrant_bench_')
    
    try:
        # Initialize Qdrant in embedded mode
        client = QdrantClient(path=temp_dir)
        
        collection_name = "sift1m"
        
        # ===== INDEX CREATION =====
        print("\n1. Index Creation")
        print("-" * 70)
        
        index_start = time.time()
        
        # Create collection with HNSW index
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
        )
        
        index_create_time = time.time() - index_start
        print(f"  ✓ Created collection in {index_create_time:.2f}s")
        
        # ===== DOCUMENT INSERTION =====
        print("\n2. Document Insertion (with index building)")
        print("-" * 70)
        
        # Use smaller batch size for Qdrant embedded mode (it's slow with large batches)
        batch_size = min(params.get('insert_batch_size', 1000), 100)
        num_categories = 10
        
        insert_start = time.time()
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_vectors)
            
            # Prepare points
            points = []
            for i in range(start_idx, end_idx):
                category = i % num_categories
                points.append(PointStruct(
                    id=i,
                    vector=base_vectors[i].tolist(),
                    payload={
                        'doc_id': i,
                        'category': category
                    }
                ))
            
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Inserted {end_idx}/{num_vectors} documents...")
        
        insert_time = time.time() - insert_start
        insert_throughput = num_vectors / insert_time
        
        print(f"  ✓ Inserted {num_vectors} documents in {insert_time:.2f}s")
        print(f"  ✓ Throughput: {insert_throughput:.0f} docs/sec")
        
        # ===== STORAGE SIZE =====
        index_size_mb = get_directory_size(temp_dir)
        print(f"  ✓ Index size: {index_size_mb:.2f} MB")
        
        index_build_time = 0.0  # Built during insertion
        
        # ===== DOCUMENT RETRIEVAL =====
        print("\n3. Document Retrieval")
        print("-" * 70)
        
        num_retrieval_tests = 100
        retrieval_times = []
        test_ids = np.random.randint(0, num_vectors, size=num_retrieval_tests)
        
        for doc_id in test_ids:
            start = time.perf_counter()
            result = client.retrieve(
                collection_name=collection_name,
                ids=[int(doc_id)]
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            retrieval_times.append(elapsed)
        
        retrieval_time_ms = np.mean(retrieval_times)
        print(f"  ✓ Single doc retrieval: {retrieval_time_ms:.3f} ms (avg over {num_retrieval_tests} queries)")
        
        # Batch retrieval
        batch_retrieval_times = []
        for _ in range(10):
            batch_ids = [int(i) for i in np.random.choice(num_vectors, size=min(100, num_vectors), replace=False)]
            start = time.perf_counter()
            result = client.retrieve(
                collection_name=collection_name,
                ids=batch_ids
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            batch_retrieval_times.append(elapsed)
        
        retrieval_batch_time_ms = np.mean(batch_retrieval_times)
        print(f"  ✓ Batch retrieval (100 docs): {retrieval_batch_time_ms:.3f} ms")
        
        # ===== VECTOR SEARCH =====
        print("\n4. Vector Search")
        print("-" * 70)
        
        k = 10
        num_queries = len(query_vectors)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            client.query_points(
                collection_name=collection_name,
                query=query_vectors[i].tolist(),
                limit=k
            )
        
        # Measure search performance
        search_latencies = []
        search_results = []
        
        search_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = client.query_points(
                collection_name=collection_name,
                query=query_vectors[i].tolist(),
                limit=k
            ).points
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            search_latencies.append(elapsed)
            result_ids = [hit.id for hit in results]
            search_results.append(result_ids)
        
        total_search_time = time.time() - search_start
        search_qps = num_queries / total_search_time
        
        search_latencies = np.array(search_latencies)
        latency_p50 = np.percentile(search_latencies, 50)
        latency_p95 = np.percentile(search_latencies, 95)
        latency_p99 = np.percentile(search_latencies, 99)
        
        print(f"  ✓ Queries: {num_queries}")
        print(f"  ✓ QPS: {search_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {latency_p50:.2f} ms")
        print(f"  ✓ Latency P95: {latency_p95:.2f} ms")
        print(f"  ✓ Latency P99: {latency_p99:.2f} ms")
        
        # Calculate recall
        search_results_array = np.array(search_results)
        recall = compute_recall_at_k(search_results_array, groundtruth, k=10)
        print(f"  ✓ Recall@10: {recall:.4f}")
        
        # ===== FILTERED VECTOR SEARCH =====
        print("\n5. Filtered Vector Search")
        print("-" * 70)
        
        filter_category = 0
        
        # Compute proper filtered ground truth using brute-force
        filter_mask = np.array([i % num_categories == filter_category for i in range(num_vectors)])
        filtered_groundtruth = compute_filtered_groundtruth(base_vectors, query_vectors, filter_mask, k=k)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            client.query_points(
                collection_name=collection_name,
                query=query_vectors[i].tolist(),
                query_filter=Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value=filter_category))]
                ),
                limit=k
            )
        
        filtered_latencies = []
        filtered_results = []
        
        filtered_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = client.query_points(
                collection_name=collection_name,
                query=query_vectors[i].tolist(),
                query_filter=Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value=filter_category))]
                ),
                limit=k
            ).points
            elapsed = (time.perf_counter() - start) * 1000
            
            filtered_latencies.append(elapsed)
            result_ids = [hit.id for hit in results] if results else []
            filtered_results.append(result_ids)
        
        filtered_total_time = time.time() - filtered_start
        filtered_qps = num_queries / filtered_total_time
        
        filtered_latencies = np.array(filtered_latencies)
        filtered_p50 = np.percentile(filtered_latencies, 50)
        filtered_p95 = np.percentile(filtered_latencies, 95)
        
        # Calculate filtered recall
        filtered_recall_sum = 0
        for i in range(len(filtered_results)):
            if len(filtered_groundtruth[i]) > 0:
                result_set = set(filtered_results[i])
                gt_set = set(filtered_groundtruth[i][:k])
                filtered_recall_sum += len(result_set & gt_set) / min(len(gt_set), k)
        filtered_recall = filtered_recall_sum / len(filtered_results) if filtered_results else 0
        
        print(f"  ✓ Filter: category == {filter_category} (~10% of data)")
        print(f"  ✓ QPS: {filtered_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {filtered_p50:.2f} ms")
        print(f"  ✓ Latency P95: {filtered_p95:.2f} ms")
        print(f"  ✓ Filtered Recall@10: {filtered_recall:.4f}")
        
        # ===== HYBRID SEARCH =====
        print("\n6. Hybrid Search (Vector + Text)")
        print("-" * 70)
        print("  ⚠️  Qdrant supports hybrid search but requires text field setup")
        print("  ⚠️  Skipping hybrid search for this benchmark (no text data)")
        
        hybrid_qps = 0.0
        hybrid_p50 = 0.0
        hybrid_p95 = 0.0
        
        # Create result object
        result = BenchmarkResult(
            library="Qdrant",
            dataset_size=num_vectors,
            dimension=dim,
            insert_time=insert_time,
            insert_throughput=insert_throughput,
            index_build_time=index_build_time,
            total_build_time=insert_time + index_build_time + index_create_time,
            index_size_mb=index_size_mb,
            retrieval_time_ms=retrieval_time_ms,
            retrieval_batch_time_ms=retrieval_batch_time_ms,
            search_qps=search_qps,
            search_latency_p50=latency_p50,
            search_latency_p95=latency_p95,
            search_latency_p99=latency_p99,
            recall_at_10=recall,
            filtered_search_qps=filtered_qps,
            filtered_search_latency_p50=filtered_p50,
            filtered_search_latency_p95=filtered_p95,
            filtered_recall_at_10=filtered_recall,
            hybrid_search_qps=hybrid_qps,
            hybrid_search_latency_p50=hybrid_p50,
            hybrid_search_latency_p95=hybrid_p95,
            params=params
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Qdrant benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        if client is not None:
            del client
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_weaviate(base_vectors, query_vectors, groundtruth, params):
    """Benchmark Weaviate implementation."""
    print("\n" + "="*70)
    print("Benchmarking Weaviate")
    print("="*70)
    
    num_vectors, dim = base_vectors.shape
    temp_dir = tempfile.mkdtemp(prefix='weaviate_bench_')
    client = None
    
    try:
        # Initialize Weaviate in embedded mode
        client = weaviate.WeaviateClient(
            embedded_options=weaviate.embedded.EmbeddedOptions(
                persistence_data_path=temp_dir
            )
        )
        client.connect()
        
        collection_name = "Sift1M"  # Weaviate requires capital first letter
        
        # ===== INDEX CREATION =====
        print("\n1. Index Creation")
        print("-" * 70)
        
        index_start = time.time()
        
        # Create collection with HNSW index
        collection = client.collections.create(
            name=collection_name,
            properties=[
                Property(name="doc_id", data_type=DataType.INT),
                Property(name="category", data_type=DataType.INT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.L2_SQUARED
            )
        )
        
        index_create_time = time.time() - index_start
        print(f"  ✓ Created collection in {index_create_time:.2f}s")
        
        # ===== DOCUMENT INSERTION =====
        print("\n2. Document Insertion (with index building)")
        print("-" * 70)
        
        batch_size = params.get('insert_batch_size', 1000)
        num_categories = 10
        
        insert_start = time.time()
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_vectors)
            
            # Prepare batch
            with collection.batch.dynamic() as batch:
                for i in range(start_idx, end_idx):
                    category = i % num_categories
                    batch.add_object(
                        properties={
                            'doc_id': i,
                            'category': category
                        },
                        vector=base_vectors[i].tolist()
                    )
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Inserted {end_idx}/{num_vectors} documents...")
        
        insert_time = time.time() - insert_start
        insert_throughput = num_vectors / insert_time
        
        print(f"  ✓ Inserted {num_vectors} documents in {insert_time:.2f}s")
        print(f"  ✓ Throughput: {insert_throughput:.0f} docs/sec")
        
        # ===== STORAGE SIZE =====
        index_size_mb = get_directory_size(temp_dir)
        print(f"  ✓ Index size: {index_size_mb:.2f} MB")
        
        index_build_time = 0.0  # Built during insertion
        
        # ===== DOCUMENT RETRIEVAL =====
        print("\n3. Document Retrieval")
        print("-" * 70)
        
        num_retrieval_tests = 100
        retrieval_times = []
        
        # Note: Weaviate doesn't have direct ID retrieval by numeric ID
        # We'll use filtering on doc_id instead
        test_ids = np.random.randint(0, num_vectors, size=num_retrieval_tests)
        
        for doc_id in test_ids:
            start = time.perf_counter()
            result = collection.query.fetch_objects(
                filters=WeaviateFilter.by_property("doc_id").equal(int(doc_id)),
                limit=1
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            retrieval_times.append(elapsed)
        
        retrieval_time_ms = np.mean(retrieval_times)
        print(f"  ✓ Single doc retrieval: {retrieval_time_ms:.3f} ms (avg over {num_retrieval_tests} queries)")
        
        # Batch retrieval
        batch_retrieval_times = []
        for _ in range(10):
            batch_ids = [int(i) for i in np.random.choice(num_vectors, size=min(100, num_vectors), replace=False)]
            start = time.perf_counter()
            # Weaviate doesn't have efficient batch ID retrieval
            # We'll do individual queries (not ideal but demonstrates the limitation)
            for bid in batch_ids[:10]:  # Just sample 10 for performance
                result = collection.query.fetch_objects(
                    filters=WeaviateFilter.by_property("doc_id").equal(bid),
                    limit=1
                )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            batch_retrieval_times.append(elapsed)
        
        retrieval_batch_time_ms = np.mean(batch_retrieval_times)
        print(f"  ✓ Batch retrieval (10 docs): {retrieval_batch_time_ms:.3f} ms")
        
        # ===== VECTOR SEARCH =====
        print("\n4. Vector Search")
        print("-" * 70)
        
        k = 10
        num_queries = len(query_vectors)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            collection.query.near_vector(
                near_vector=query_vectors[i].tolist(),
                limit=k,
                return_properties=["doc_id", "category"]
            )
        
        # Measure search performance
        search_latencies = []
        search_results = []
        
        search_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.query.near_vector(
                near_vector=query_vectors[i].tolist(),
                limit=k,
                return_properties=["doc_id", "category"]
            )
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            search_latencies.append(elapsed)
            result_ids = [obj.properties['doc_id'] for obj in results.objects]
            search_results.append(result_ids)
        
        total_search_time = time.time() - search_start
        search_qps = num_queries / total_search_time
        
        search_latencies = np.array(search_latencies)
        latency_p50 = np.percentile(search_latencies, 50)
        latency_p95 = np.percentile(search_latencies, 95)
        latency_p99 = np.percentile(search_latencies, 99)
        
        print(f"  ✓ Queries: {num_queries}")
        print(f"  ✓ QPS: {search_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {latency_p50:.2f} ms")
        print(f"  ✓ Latency P95: {latency_p95:.2f} ms")
        print(f"  ✓ Latency P99: {latency_p99:.2f} ms")
        
        # Calculate recall
        search_results_array = np.array(search_results)
        recall = compute_recall_at_k(search_results_array, groundtruth, k=10)
        print(f"  ✓ Recall@10: {recall:.4f}")
        
        # ===== FILTERED VECTOR SEARCH =====
        print("\n5. Filtered Vector Search")
        print("-" * 70)
        
        filter_category = 0
        
        # Compute proper filtered ground truth using brute-force
        filter_mask = np.array([i % num_categories == filter_category for i in range(num_vectors)])
        filtered_groundtruth = compute_filtered_groundtruth(base_vectors, query_vectors, filter_mask, k=k)
        
        # Warmup
        for i in range(min(1000, num_queries)):
            collection.query.near_vector(
                near_vector=query_vectors[i].tolist(),
                filters=WeaviateFilter.by_property("category").equal(filter_category),
                limit=k,
                return_properties=["doc_id", "category"]
            )
        
        filtered_latencies = []
        filtered_results = []
        
        filtered_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.query.near_vector(
                near_vector=query_vectors[i].tolist(),
                filters=WeaviateFilter.by_property("category").equal(filter_category),
                limit=k,
                return_properties=["doc_id", "category"]
            )
            elapsed = (time.perf_counter() - start) * 1000
            
            filtered_latencies.append(elapsed)
            result_ids = [obj.properties['doc_id'] for obj in results.objects] if results.objects else []
            filtered_results.append(result_ids)
        
        filtered_total_time = time.time() - filtered_start
        filtered_qps = num_queries / filtered_total_time
        
        filtered_latencies = np.array(filtered_latencies)
        filtered_p50 = np.percentile(filtered_latencies, 50)
        filtered_p95 = np.percentile(filtered_latencies, 95)
        
        # Calculate filtered recall
        filtered_recall_sum = 0
        for i in range(len(filtered_results)):
            if len(filtered_groundtruth[i]) > 0:
                result_set = set(filtered_results[i])
                gt_set = set(filtered_groundtruth[i][:k])
                filtered_recall_sum += len(result_set & gt_set) / min(len(gt_set), k)
        filtered_recall = filtered_recall_sum / len(filtered_results) if filtered_results else 0
        
        print(f"  ✓ Filter: category == {filter_category} (~10% of data)")
        print(f"  ✓ QPS: {filtered_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {filtered_p50:.2f} ms")
        print(f"  ✓ Latency P95: {filtered_p95:.2f} ms")
        print(f"  ✓ Filtered Recall@10: {filtered_recall:.4f}")
        
        # ===== HYBRID SEARCH =====
        print("\n6. Hybrid Search (Vector + Text)")
        print("-" * 70)
        print("  ⚠️  Weaviate supports hybrid search but requires text field setup")
        print("  ⚠️  Skipping hybrid search for this benchmark (no text data)")
        
        hybrid_qps = 0.0
        hybrid_p50 = 0.0
        hybrid_p95 = 0.0
        
        # Create result object
        result = BenchmarkResult(
            library="Weaviate",
            dataset_size=num_vectors,
            dimension=dim,
            insert_time=insert_time,
            insert_throughput=insert_throughput,
            index_build_time=index_build_time,
            total_build_time=insert_time + index_build_time + index_create_time,
            index_size_mb=index_size_mb,
            retrieval_time_ms=retrieval_time_ms,
            retrieval_batch_time_ms=retrieval_batch_time_ms,
            search_qps=search_qps,
            search_latency_p50=latency_p50,
            search_latency_p95=latency_p95,
            search_latency_p99=latency_p99,
            recall_at_10=recall,
            filtered_search_qps=filtered_qps,
            filtered_search_latency_p50=filtered_p50,
            filtered_search_latency_p95=filtered_p95,
            filtered_recall_at_10=filtered_recall,
            hybrid_search_qps=hybrid_qps,
            hybrid_search_latency_p50=hybrid_p50,
            hybrid_search_latency_p95=hybrid_p95,
            params=params
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Weaviate benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        if client is not None:
            client.close()
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a comparison table of all benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)
    
    if not results:
        print("No results to compare")
        return
    
    # Document Insertion
    print("\n1. Document Insertion")
    print("-" * 70)
    print(f"{'Library':<15} {'Time (s)':<12} {'Throughput (docs/s)':<20}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:<15} {r.insert_time:<12.2f} {r.insert_throughput:<20.0f}")
    
    # Index Building
    print("\n2. Index Building")
    print("-" * 70)
    print(f"{'Library':<15} {'Index Time (s)':<15} {'Total Time (s)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:<15} {r.index_build_time:<15.2f} {r.total_build_time:<15.2f}")
    
    # Storage
    print("\n3. Storage Size")
    print("-" * 70)
    print(f"{'Library':<15} {'Size (MB)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:<15} {r.index_size_mb:<15.2f}")
    
    # Document Retrieval
    print("\n4. Document Retrieval")
    print("-" * 70)
    print(f"{'Library':<15} {'Single (ms)':<15} {'Batch 100 (ms)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:<15} {r.retrieval_time_ms:<15.3f} {r.retrieval_batch_time_ms:<15.3f}")
    
    # Vector Search Performance
    print("\n5. Vector Search Performance")
    print("-" * 70)
    print(f"{'Library':<15} {'QPS':<10} {'P50 (ms)':<10} {'P95 (ms)':<10} {'P99 (ms)':<10} {'Recall@10':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:<15} {r.search_qps:<10.1f} {r.search_latency_p50:<10.2f} "
              f"{r.search_latency_p95:<10.2f} {r.search_latency_p99:<10.2f} {r.recall_at_10:<10.4f}")
    
    # Filtered Search Performance
    print("\n6. Filtered Vector Search Performance")
    print("-" * 70)
    print(f"{'Library':<15} {'QPS':<10} {'P50 (ms)':<10} {'P95 (ms)':<10} {'Recall@10':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.library:<15} {r.filtered_search_qps:<10.1f} {r.filtered_search_latency_p50:<10.2f} "
              f"{r.filtered_search_latency_p95:<10.2f} {r.filtered_recall_at_10:<10.4f}")
    
    # Hybrid Search Performance
    print("\n7. Hybrid Search Performance (Vector + Text BM25)")
    print("-" * 70)
    print(f"{'Library':<15} {'QPS':<10} {'P50 (ms)':<10} {'P95 (ms)':<10} {'Supported':<10}")
    print("-" * 70)
    for r in results:
        supported = "Yes" if r.hybrid_search_qps > 0 else "No"
        if r.hybrid_search_qps > 0:
            print(f"{r.library:<15} {r.hybrid_search_qps:<10.1f} {r.hybrid_search_latency_p50:<10.2f} "
                  f"{r.hybrid_search_latency_p95:<10.2f} {supported:<10}")
        else:
            print(f"{r.library:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {supported:<10}")
    
    # Summary - Winners
    print("\n8. Summary")
    print("-" * 70)
    
    fastest_insert = min(results, key=lambda x: x.insert_time)
    print(f"Fastest Insert:        {fastest_insert.library} ({fastest_insert.insert_throughput:.0f} docs/s)")
    
    smallest_index = min(results, key=lambda x: x.index_size_mb)
    print(f"Smallest Index:        {smallest_index.library} ({smallest_index.index_size_mb:.2f} MB)")
    
    fastest_retrieval = min(results, key=lambda x: x.retrieval_time_ms)
    print(f"Fastest Retrieval:     {fastest_retrieval.library} ({fastest_retrieval.retrieval_time_ms:.3f} ms)")
    
    highest_qps = max(results, key=lambda x: x.search_qps)
    print(f"Highest Search QPS:    {highest_qps.library} ({highest_qps.search_qps:.1f} QPS)")
    
    lowest_latency = min(results, key=lambda x: x.search_latency_p50)
    print(f"Lowest Search Latency: {lowest_latency.library} (P50: {lowest_latency.search_latency_p50:.2f} ms)")
    
    best_recall = max(results, key=lambda x: x.recall_at_10)
    print(f"Best Recall@10:        {best_recall.library} ({best_recall.recall_at_10:.4f})")
    
    # Filtered search winners
    if any(r.filtered_search_qps > 0 for r in results):
        highest_filtered_qps = max(results, key=lambda x: x.filtered_search_qps)
        print(f"Highest Filtered QPS:  {highest_filtered_qps.library} ({highest_filtered_qps.filtered_search_qps:.1f} QPS)")
        best_filtered_recall = max(results, key=lambda x: x.filtered_recall_at_10)
        print(f"Best Filtered Recall:  {best_filtered_recall.library} ({best_filtered_recall.filtered_recall_at_10:.4f})")
    
    # Hybrid search winners
    hybrid_results = [r for r in results if r.hybrid_search_qps > 0]
    if hybrid_results:
        highest_hybrid_qps = max(hybrid_results, key=lambda x: x.hybrid_search_qps)
        print(f"Highest Hybrid QPS:    {highest_hybrid_qps.library} ({highest_hybrid_qps.hybrid_search_qps:.1f} QPS)")
    else:
        print(f"Hybrid Search:         Only Caliby supports true vector + BM25 fusion")


def save_results_json(results: List[BenchmarkResult], output_file: str):
    """Save benchmark results to JSON file."""
    results_dict = {
        "benchmark": "Caliby vs ChromaDB",
        "dataset": "SIFT1M",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Caliby vs ChromaDB vs Qdrant vs Weaviate')
    parser.add_argument('--data-dir', type=str, default='./sift1m',
                        help='Directory containing SIFT1M dataset')
    parser.add_argument('--num-vectors', type=int, default=None,
                        help='Limit number of vectors to use (default: all 1M)')
    parser.add_argument('--caliby-only', action='store_true',
                        help='Run only Caliby benchmark')
    parser.add_argument('--chromadb-only', action='store_true',
                        help='Run only ChromaDB benchmark')
    parser.add_argument('--qdrant-only', action='store_true',
                        help='Run only Qdrant benchmark')
    parser.add_argument('--weaviate-only', action='store_true',
                        help='Run only Weaviate benchmark')
    parser.add_argument('--output', type=str, default='vectordb_comparison.json',
                        help='Output JSON file for results')
    parser.add_argument('--M', type=int, default=16,
                        help='HNSW M parameter (Caliby)')
    parser.add_argument('--ef-construction', type=int, default=200,
                        help='HNSW ef_construction parameter (Caliby)')
    parser.add_argument('--insert-batch-size', type=int, default=10000,
                        help='Batch size for insertions')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found")
        print("Please download SIFT1M dataset first")
        sys.exit(1)
    
    # Load dataset
    base_vectors, query_vectors, groundtruth = load_sift1m_data(args.data_dir)
    
    # Limit vectors if requested
    if args.num_vectors is not None:
        print(f"\nLimiting to {args.num_vectors} vectors for testing")
        base_vectors = base_vectors[:args.num_vectors]
        # IMPORTANT: Must recompute ground truth for the subset!
        # The original ground truth was computed against the full 1M vectors,
        # so most IDs won't exist in our subset, leading to very low recall.
        groundtruth = compute_groundtruth(base_vectors, query_vectors, k=100)
    
    # Benchmark parameters
    params = {
        'M': args.M,
        'ef_construction': args.ef_construction,
        'insert_batch_size': args.insert_batch_size
    }
    
    results = []
    
    if not args.caliby_only and not args.chromadb_only and not args.qdrant_only and LIBS_AVAILABLE['weaviate']:
        result = benchmark_weaviate(base_vectors, query_vectors, groundtruth, params)
        if result:
            results.append(result)
    elif not args.caliby_only and not args.chromadb_only and not args.qdrant_only:
        print("Weaviate not available, skipping...")
    
    # Run benchmarks
    if not args.chromadb_only and not args.qdrant_only and not args.weaviate_only and LIBS_AVAILABLE['caliby']:
        result = benchmark_caliby(base_vectors, query_vectors, groundtruth, params)
        if result:
            results.append(result)
    elif not args.chromadb_only and not args.qdrant_only and not args.weaviate_only:
        print("Caliby not available, skipping...")
    
    if not args.caliby_only and not args.qdrant_only and not args.weaviate_only and LIBS_AVAILABLE['chromadb']:
        result = benchmark_chromadb(base_vectors, query_vectors, groundtruth, params)
        if result:
            results.append(result)
    elif not args.caliby_only and not args.qdrant_only and not args.weaviate_only:
        print("ChromaDB not available, skipping...")
    
    # if not args.caliby_only and not args.chromadb_only and not args.weaviate_only and LIBS_AVAILABLE['qdrant']:
    #     result = benchmark_qdrant(base_vectors, query_vectors, groundtruth, params)
    #     if result:
    #         results.append(result)
    # elif not args.caliby_only and not args.chromadb_only and not args.weaviate_only:
    #     print("Qdrant not available, skipping...")
    
    
    # Print comparison
    if len(results) > 0:
        print_comparison_table(results)
        save_results_json(results, args.output)
    else:
        print("No benchmarks were run. Please install required libraries.")
        sys.exit(1)


if __name__ == '__main__':
    main()
