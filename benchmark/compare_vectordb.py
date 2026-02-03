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
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchParams
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


def read_fbin(filename):
    """Read .fbin file format (used by deep10M dataset).
    
    Format: [num_vectors (4 bytes)][dim (4 bytes)][float32 data...]
    """
    with open(filename, 'rb') as f:
        num_vectors = struct.unpack('i', f.read(4))[0]
        dim = struct.unpack('i', f.read(4))[0]
        # Read all float data at once
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        return data.reshape(num_vectors, dim)


def read_ibin(filename):
    """Read .ibin file format (used for deep10M ground truth).
    
    Format: [num_queries (4 bytes)][k (4 bytes)][int32 data...]
    """
    with open(filename, 'rb') as f:
        num_queries = struct.unpack('i', f.read(4))[0]
        k = struct.unpack('i', f.read(4))[0]
        # Read all int data at once
        data = np.fromfile(f, dtype=np.int32, count=num_queries * k)
        return data.reshape(num_queries, k)


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


def load_deep10m_data(data_dir='./deep10M'):
    """Load Deep10M dataset (10M 96-dimensional vectors)."""
    print("\nLoading Deep10M dataset...")
    base_vectors = read_fbin(os.path.join(data_dir, 'base.10M.fbin'))
    query_vectors = read_fbin(os.path.join(data_dir, 'query.public.10K.fbin'))
    groundtruth = read_ibin(os.path.join(data_dir, 'groundtruth.ibin'))
    
    print(f"  Base vectors: {base_vectors.shape}")
    print(f"  Query vectors: {query_vectors.shape}")
    print(f"  Ground truth: {groundtruth.shape}")
    
    return base_vectors, query_vectors, groundtruth


def load_qdrant_filtered_dataset(data_dir, dataset_name):
    """
    Load datasets from qdrant/ann-filtering-benchmark-datasets format.
    
    These datasets include:
    - vectors.npy: Base vectors
    - payloads.jsonl: Metadata for each vector
    - tests.jsonl: Test queries with filter conditions and ground truth
    - filters.json: Filter field definitions
    
    Returns:
        base_vectors: np.ndarray of shape (num_vectors, dim)
        test_queries: List of dicts with 'query', 'conditions', 'closest_ids', 'closest_scores'
        payloads: List of dicts with metadata for each vector
    """
    print(f"\nLoading {dataset_name} filtered dataset...")
    
    # Load vectors
    vectors_path = os.path.join(data_dir, 'vectors.npy')
    base_vectors = np.load(vectors_path).astype(np.float32)
    print(f"  Base vectors: {base_vectors.shape}")
    
    # Load payloads (metadata)
    payloads_path = os.path.join(data_dir, 'payloads.jsonl')
    payloads = []
    with open(payloads_path, 'r') as f:
        for line in f:
            payloads.append(json.loads(line))
    print(f"  Payloads: {len(payloads)} records")
    
    # Load test queries with filter conditions
    tests_path = os.path.join(data_dir, 'tests.jsonl')
    test_queries = []
    with open(tests_path, 'r') as f:
        for line in f:
            test_queries.append(json.loads(line))
    print(f"  Test queries: {len(test_queries)} queries")
    
    # Load filter definitions (optional, for reference)
    filters_path = os.path.join(data_dir, 'filters.json')
    if os.path.exists(filters_path):
        with open(filters_path, 'r') as f:
            filters_def = json.load(f)
        print(f"  Filter fields: {len(filters_def)} field definitions")
    
    return base_vectors, test_queries, payloads


def convert_qdrant_conditions_to_caliby(conditions, array_fields=None):
    """
    Convert Qdrant filter conditions to Caliby's filter format.
    
    Qdrant format:
    {
        "and": [
            {"field_name": {"match": {"value": "some_value"}}},
            {"field_name": {"range": {"gte": 10, "lte": 100}}}
        ]
    }
    
    Caliby format (MongoDB-style):
    {
        "$and": [
            {"field_name": "some_value"},  # for equality match
            {"field_name": {"$contains": "some_value"}},  # for array contains (if field is array)
            {"field_name": {"$gte": 10, "$lte": 100}}  # for range
        ]
    }
    
    Args:
        conditions: Qdrant filter conditions dict
        array_fields: Set of field names that are STRING_ARRAY type (use $contains for these)
    """
    if not conditions:
        return None
    
    if array_fields is None:
        array_fields = set()
    
    caliby_conditions = []
    
    if 'and' in conditions:
        for cond in conditions['and']:
            for field, op_dict in cond.items():
                if 'match' in op_dict:
                    # Equality match - but use $contains for array fields!
                    value = op_dict['match'].get('value')
                    if field in array_fields:
                        # Array field: use $contains operator
                        caliby_conditions.append({field: {"$contains": value}})
                    else:
                        # Regular field: use equality
                        caliby_conditions.append({field: value})
                elif 'range' in op_dict:
                    # Range condition
                    range_cond = {}
                    range_op = op_dict['range']
                    if 'gte' in range_op:
                        range_cond['$gte'] = range_op['gte']
                    if 'gt' in range_op:
                        range_cond['$gt'] = range_op['gt']
                    if 'lte' in range_op:
                        range_cond['$lte'] = range_op['lte']
                    if 'lt' in range_op:
                        range_cond['$lt'] = range_op['lt']
                    caliby_conditions.append({field: range_cond})
                elif 'geo' in op_dict:
                    # Geo conditions not supported yet, skip
                    print(f"Warning: geo conditions not supported, skipping")
                    continue
    
    if len(caliby_conditions) == 0:
        return None
    elif len(caliby_conditions) == 1:
        return caliby_conditions[0]
    else:
        return {"$and": caliby_conditions}


def convert_qdrant_conditions_to_chromadb(conditions):
    """
    Convert Qdrant filter conditions to ChromaDB's where format.
    
    ChromaDB format:
    {"field_name": {"$eq": "some_value"}}
    or for multiple conditions:
    {"$and": [{"field_name": {"$eq": "some_value"}}, ...]}
    """
    if not conditions:
        return None
    
    chromadb_conditions = []
    
    if 'and' in conditions:
        for cond in conditions['and']:
            for field, op_dict in cond.items():
                if 'match' in op_dict:
                    value = op_dict['match'].get('value')
                    chromadb_conditions.append({field: {"$eq": value}})
                elif 'range' in op_dict:
                    range_op = op_dict['range']
                    range_conds = []
                    if 'gte' in range_op:
                        range_conds.append({field: {"$gte": range_op['gte']}})
                    if 'gt' in range_op:
                        range_conds.append({field: {"$gt": range_op['gt']}})
                    if 'lte' in range_op:
                        range_conds.append({field: {"$lte": range_op['lte']}})
                    if 'lt' in range_op:
                        range_conds.append({field: {"$lt": range_op['lt']}})
                    chromadb_conditions.extend(range_conds)
    
    if len(chromadb_conditions) == 0:
        return None
    elif len(chromadb_conditions) == 1:
        return chromadb_conditions[0]
    else:
        return {"$and": chromadb_conditions}


def convert_qdrant_conditions_to_qdrant(conditions):
    """
    Convert Qdrant benchmark conditions to qdrant-client Filter format.
    Returns a Filter object for use with qdrant-client.
    """
    if not conditions or 'and' not in conditions:
        return None
    
    must_conditions = []
    
    for cond in conditions['and']:
        for field, op_dict in cond.items():
            if 'match' in op_dict:
                value = op_dict['match'].get('value')
                must_conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
            elif 'range' in op_dict:
                # Qdrant uses Range model for range conditions
                from qdrant_client.models import Range
                range_op = op_dict['range']
                must_conditions.append(FieldCondition(
                    key=field, 
                    range=Range(
                        gte=range_op.get('gte'),
                        gt=range_op.get('gt'),
                        lte=range_op.get('lte'),
                        lt=range_op.get('lt')
                    )
                ))
    
    if len(must_conditions) == 0:
        return None
    
    return Filter(must=must_conditions)


def convert_qdrant_conditions_to_weaviate(conditions):
    """
    Convert Qdrant benchmark conditions to Weaviate Filter format.
    """
    if not conditions or 'and' not in conditions:
        return None
    
    filters = []
    
    for cond in conditions['and']:
        for field, op_dict in cond.items():
            if 'match' in op_dict:
                value = op_dict['match'].get('value')
                filters.append(WeaviateFilter.by_property(field).equal(value))
            elif 'range' in op_dict:
                range_op = op_dict['range']
                if 'gte' in range_op:
                    filters.append(WeaviateFilter.by_property(field).greater_or_equal(range_op['gte']))
                if 'gt' in range_op:
                    filters.append(WeaviateFilter.by_property(field).greater_than(range_op['gt']))
                if 'lte' in range_op:
                    filters.append(WeaviateFilter.by_property(field).less_or_equal(range_op['lte']))
                if 'lt' in range_op:
                    filters.append(WeaviateFilter.by_property(field).less_than(range_op['lt']))
    
    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        # Combine with AND
        result = filters[0]
        for f in filters[1:]:
            result = result & f
        return result


def compute_groundtruth(base_vectors, query_vectors, k=100, metric='l2'):
    """
    Compute ground truth by brute-force distance calculation.
    
    This is necessary when using a subset of the base vectors, because
    the original ground truth was computed on the full dataset.
    
    Args:
        base_vectors: Shape (n_base, dim)
        query_vectors: Shape (n_queries, dim)
        k: Number of nearest neighbors to find
        metric: Distance metric to use - 'l2' for L2 distance or 'cosine' for cosine distance
    
    Returns:
        Ground truth array of shape (n_queries, k) with indices
    """
    print(f"\nComputing ground truth ({metric}) for {len(query_vectors)} queries against {len(base_vectors)} vectors...")
    n_queries = len(query_vectors)
    n_base = len(base_vectors)
    
    # For larger datasets, process in batches to manage memory
    groundtruth = np.zeros((n_queries, k), dtype=np.int32)
    
    # Pre-compute normalized vectors for cosine similarity
    if metric == 'cosine':
        base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
        base_normalized = base_vectors / (base_norms + 1e-10)
    
    batch_size = 100  # Process queries in batches
    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        batch_queries = query_vectors[batch_start:batch_end]
        
        if metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            # Cosine similarity = dot(q_norm, b_norm)
            query_norms = np.linalg.norm(batch_queries, axis=1, keepdims=True)
            query_normalized = batch_queries / (query_norms + 1e-10)
            similarities = np.dot(query_normalized, base_normalized.T)
            # We want smallest distance, so negate similarity
            distances = -similarities
        else:
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


def compute_filtered_groundtruth(base_vectors, query_vectors, filter_mask, k=10, metric='l2'):
    """
    Compute ground truth for filtered search by brute-force.
    
    Args:
        base_vectors: Shape (n_base, dim)
        query_vectors: Shape (n_queries, dim)
        filter_mask: Boolean array of shape (n_base,) where True means the vector passes the filter
        k: Number of nearest neighbors to find
        metric: Distance metric to use - 'l2' for L2 distance or 'cosine' for cosine distance
    
    Returns:
        List of lists, where each inner list contains the k nearest neighbor indices
        that pass the filter for that query
    """
    print(f"\nComputing filtered ground truth ({metric}) for {len(query_vectors)} queries...")
    n_queries = len(query_vectors)
    n_base = len(base_vectors)
    
    # Get indices that pass the filter
    filtered_indices = np.where(filter_mask)[0]
    filtered_vectors = base_vectors[filtered_indices]
    n_filtered = len(filtered_indices)
    
    print(f"  Filtered to {n_filtered} vectors ({100*n_filtered/n_base:.1f}% of data)")
    
    if n_filtered == 0:
        return [[] for _ in range(n_queries)]
    
    # Pre-compute normalized vectors for cosine similarity
    if metric == 'cosine':
        filtered_norms = np.linalg.norm(filtered_vectors, axis=1, keepdims=True)
        filtered_normalized = filtered_vectors / (filtered_norms + 1e-10)
    
    filtered_groundtruth = []
    
    batch_size = 100  # Process queries in batches
    for batch_start in range(0, n_queries, batch_size):
        batch_end = min(batch_start + batch_size, n_queries)
        batch_queries = query_vectors[batch_start:batch_end]
        
        if metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            query_norms = np.linalg.norm(batch_queries, axis=1, keepdims=True)
            query_normalized = batch_queries / (query_norms + 1e-10)
            similarities = np.dot(query_normalized, filtered_normalized.T)
            distances = -similarities  # Negate for argpartition (smaller is better)
        else:
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
            collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, ef_search=100)
        
        # Measure search performance
        search_latencies = []
        search_results = []
        
        search_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, ef_search=100)
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
            collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, filter=filter_json, ef_search=100)
        
        filtered_latencies = []
        filtered_results = []
        
        filtered_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, filter=filter_json, ef_search=100)
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


def benchmark_caliby_filtered(base_vectors, query_vectors, groundtruth, params):
    """
    Benchmark Caliby with filtered benchmark datasets (arxiv, hnm).
    
    These datasets have real payloads and pre-computed filtered ground truth.
    """
    print("\n" + "="*70)
    print("Benchmarking Caliby (Filtered Dataset)")
    print("="*70)
    
    test_queries = params.get('test_queries', [])
    payloads = params.get('payloads', [])
    dataset_type = params.get('dataset_type', 'unknown')
    
    if not test_queries or not payloads:
        print("Error: test_queries and payloads required for filtered benchmark")
        return None
    print(f"  # Test queries: {len(test_queries)}")

    num_vectors, dim = base_vectors.shape
    temp_dir = tempfile.mkdtemp(prefix='caliby_filtered_bench_')
    
    try:
        # Initialize database
        caliby.open(temp_dir, cleanup_if_exist=True)
        
        # Extract filterable fields from test queries
        filterable_fields = set()
        for tq in test_queries[:100]:  # Sample first 100 queries
            conditions = tq.get('conditions', {})
            if 'and' in conditions:
                for cond in conditions['and']:
                    for field in cond.keys():
                        filterable_fields.add(field)
        
        print(f"  Filterable fields from test queries: {filterable_fields}")
        
        # Infer schema from first payload, but only for filterable fields + doc_id
        sample_payload = payloads[0]
        schema = caliby.Schema()
        schema.add_field("doc_id", caliby.FieldType.INT)
        
        # Add fields based on payload types (only filterable ones)
        metadata_fields = []
        array_fields = set()  # Track which fields are arrays
        for key, value in sample_payload.items():
            if key not in filterable_fields:
                continue  # Skip non-filterable fields
            if isinstance(value, str):
                schema.add_field(key, caliby.FieldType.STRING)
                metadata_fields.append(key)
            elif isinstance(value, (int, float)):
                schema.add_field(key, caliby.FieldType.FLOAT)
                metadata_fields.append(key)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                # List of strings (like labels) - use STRING_ARRAY for proper array matching!
                schema.add_field(key, caliby.FieldType.STRING_ARRAY)
                metadata_fields.append(key)
                array_fields.add(key)
            elif value is None:
                # Assume string type for None values
                schema.add_field(key, caliby.FieldType.STRING)
                metadata_fields.append(key)
        
        print(f"  Schema fields: {metadata_fields}")
        print(f"  Array fields (will use $contains): {array_fields}")
        
        # Use cosine distance for embedding-based datasets
        distance_metric = caliby.DistanceMetric.COSINE if dataset_type in ['arxiv', 'hnm'] else caliby.DistanceMetric.L2
        
        # Create collection
        collection = caliby.Collection(
            dataset_type,
            schema,
            vector_dim=dim,
            distance_metric=distance_metric
        )
        
        # ===== INDEX CREATION =====
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
        
        # Create metadata indexes for filterable fields
        meta_idx_start = time.time()
        for field in metadata_fields:
            try:
                collection.create_metadata_index(f"{field}_idx", [field])
            except Exception as e:
                print(f"  Warning: Could not create index for {field}: {e}")
        meta_idx_time = time.time() - meta_idx_start
        print(f"  ✓ Created metadata indexes in {meta_idx_time:.2f}s")
        
        # ===== DOCUMENT INSERTION =====
        print("\n2. Document Insertion")
        print("-" * 70)
        
        batch_size = params.get('insert_batch_size', 10000)
        
        insert_start = time.time()
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_vectors)
            
            batch_contents = [f"doc_{i}" for i in range(start_idx, end_idx)]
            batch_metadatas = []
            for i in range(start_idx, end_idx):
                # Only include filterable fields + doc_id
                payload = {'doc_id': i}
                orig_payload = payloads[i]
                for key in metadata_fields:
                    value = orig_payload.get(key)
                    if value is None:
                        if key in array_fields:
                            payload[key] = []  # Empty array for array fields
                        else:
                            payload[key] = ""
                    elif isinstance(value, list):
                        # Keep arrays as arrays for STRING_ARRAY fields!
                        if key in array_fields:
                            payload[key] = value  # Keep as list
                        else:
                            payload[key] = ','.join(str(v) for v in value)
                    else:
                        payload[key] = value
                batch_metadatas.append(payload)
            batch_vectors = base_vectors[start_idx:end_idx].tolist()
            
            collection.add(batch_contents, batch_metadatas, batch_vectors)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Inserted {end_idx}/{num_vectors} documents...")
        
        insert_time = time.time() - insert_start
        insert_throughput = num_vectors / insert_time
        
        print(f"  ✓ Inserted {num_vectors} documents in {insert_time:.2f}s")
        print(f"  ✓ Throughput: {insert_throughput:.0f} docs/sec")
        
        caliby.flush_storage()
        index_size_mb = get_directory_size(temp_dir)
        print(f"  ✓ Index size: {index_size_mb:.2f} MB")
        
        # ===== UNFILTERED VECTOR SEARCH =====
        print("\n3. Unfiltered Vector Search")
        print("-" * 70)
        
        # For filtered datasets (arxiv, hnm), the groundtruth from the dataset is filtered.
        # We need to compute proper unfiltered groundtruth using cosine distance.
        # Note: Query vectors are often copies of base vectors, so we expect perfect matches.
        metric = 'cosine' if dataset_type in ['arxiv', 'hnm'] else 'l2'
        unfiltered_groundtruth = compute_groundtruth(base_vectors, query_vectors, k=100, metric=metric)
        
        k = 10
        num_queries = len(query_vectors)
        
        # Warmup
        for i in range(min(100, num_queries)):
            collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, ef_search=100)
        
        search_latencies = []
        search_results = []
        
        search_start = time.time()
        for i in range(num_queries):
            start = time.perf_counter()
            results = collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, ef_search=100)
            elapsed = (time.perf_counter() - start) * 1000
            
            search_latencies.append(elapsed)
            search_results.append([r.doc_id for r in results])
        
        total_search_time = time.time() - search_start
        search_qps = num_queries / total_search_time
        
        search_latencies = np.array(search_latencies)
        latency_p50 = np.percentile(search_latencies, 50)
        latency_p95 = np.percentile(search_latencies, 95)
        latency_p99 = np.percentile(search_latencies, 99)
        
        # Calculate recall against properly computed unfiltered ground truth
        search_results_array = np.array(search_results)
        recall = compute_recall_at_k(search_results_array, unfiltered_groundtruth, k=10)
        
        print(f"  ✓ QPS: {search_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {latency_p50:.2f} ms")
        print(f"  ✓ Latency P95: {latency_p95:.2f} ms")
        print(f"  ✓ Recall@10: {recall:.4f}")
        
        # ===== FILTERED VECTOR SEARCH =====
        print("\n4. Filtered Vector Search (from test queries)")
        print("-" * 70)
        
        # WARNING: The filtered ground truth in test_queries was computed on the FULL dataset.
        # When using a subset (--num-vectors), recall will be artificially low because:
        # 1. Many closest_ids reference documents outside our subset range
        # 2. Filter selectivity is different in the subset
        # For accurate filtered search benchmarking, use the full dataset.
        if num_vectors < len(payloads):
            print(f"  ⚠ WARNING: Using subset of data ({num_vectors}/{len(payloads)} vectors)")
            print(f"    Filtered recall will be artificially low - ground truth was computed on full dataset")
            print(f"    For accurate filtered benchmarks, run without --num-vectors flag")
        
        filtered_latencies = []
        filtered_results = []
        filtered_groundtruths = []
        
        # Use the pre-defined test queries with filters
        num_filtered_queries = len(test_queries)
        
        # Debug: Count filter types and analyze selectivity
        filter_type_counts = {'labels': 0, 'submitter': 0, 'update_date_ts': 0, 'other': 0}
        empty_result_count = 0
        
        # Warmup with a few queries
        for i in range(min(10, num_filtered_queries)):
            tq = test_queries[i]
            conditions = tq.get('conditions', {})
            filter_json = json.dumps(convert_qdrant_conditions_to_caliby(conditions, array_fields)) if conditions else ""
            collection.search_vector(query_vectors[i].tolist(), "vec_idx", k=k, filter=filter_json, ef_search=100)
        
        filtered_start = time.time()
        skipped_contains_count = 0
        for i, tq in enumerate(test_queries):
            query_vec = tq['query']
            conditions = tq.get('conditions', {})
            expected_ids = tq.get('closest_ids', [])[:k]
            
            # Count filter types
            if 'and' in conditions:
                for cond in conditions['and']:
                    for field in cond.keys():
                        if field in filter_type_counts:
                            filter_type_counts[field] += 1
                        else:
                            filter_type_counts['other'] += 1
            
            # Convert conditions to Caliby format (pass array_fields to use $contains for arrays)
            caliby_filter = convert_qdrant_conditions_to_caliby(conditions, array_fields)
            
            # Skip queries that use $contains (array fields) for now - btree doesn't support them
            filter_json = json.dumps(caliby_filter) if caliby_filter else ""
            if '$contains' in filter_json:
                skipped_contains_count += 1
                continue
            
            start = time.perf_counter()
            results = collection.search_vector(query_vec, "vec_idx", k=k, filter=filter_json, ef_search=100)
            elapsed = (time.perf_counter() - start) * 1000
            
            if len(results) == 0:
                empty_result_count += 1
            
            filtered_latencies.append(elapsed)
            filtered_results.append([r.doc_id for r in results])
            filtered_groundtruths.append(expected_ids)
        
        if skipped_contains_count > 0:
            print(f"  (Skipped {skipped_contains_count} queries with $contains filters)")
        
        num_filtered_queries = len(filtered_latencies)  # Update count after skipping
        filtered_total_time = time.time() - filtered_start
        filtered_qps = num_filtered_queries / filtered_total_time if num_filtered_queries > 0 else 0
        
        filtered_latencies = np.array(filtered_latencies) if filtered_latencies else np.array([0])
        filtered_p50 = np.percentile(filtered_latencies, 50)
        filtered_p95 = np.percentile(filtered_latencies, 95)
        
        # Calculate filtered recall per filter type
        filter_recalls = {'labels': [], 'submitter': [], 'update_date_ts': []}
        
        # Calculate filtered recall
        filtered_recall_sum = 0
        for i in range(len(filtered_results)):
            if len(filtered_groundtruths[i]) > 0:
                result_set = set(filtered_results[i])
                gt_set = set(filtered_groundtruths[i][:k])
                query_recall = len(result_set & gt_set) / min(len(gt_set), k)
                filtered_recall_sum += query_recall
                
                # Track per-filter-type recall
                conditions = test_queries[i].get('conditions', {})
                if 'and' in conditions:
                    for cond in conditions['and']:
                        for field in cond.keys():
                            if field in filter_recalls:
                                filter_recalls[field].append(query_recall)
        
        filtered_recall = filtered_recall_sum / len(filtered_results) if filtered_results else 0
        
        print(f"  ✓ Filtered queries: {num_filtered_queries}")
        print(f"  ✓ Filter type distribution: {filter_type_counts}")
        print(f"  ✓ Empty result count: {empty_result_count} ({100*empty_result_count/num_filtered_queries:.1f}%)")
        print(f"  ✓ QPS: {filtered_qps:.1f} queries/sec")
        print(f"  ✓ Latency P50: {filtered_p50:.2f} ms")
        print(f"  ✓ Latency P95: {filtered_p95:.2f} ms")
        print(f"  ✓ Filtered Recall@10: {filtered_recall:.4f}")
        
        # Print per-filter-type recall
        for field, recalls in filter_recalls.items():
            if recalls:
                print(f"    - {field} queries ({len(recalls)}): recall={np.mean(recalls):.4f}")
        
        # Document retrieval (simple test)
        retrieval_time_ms = 0.0
        retrieval_batch_time_ms = 0.0
        
        result = BenchmarkResult(
            library="Caliby",
            dataset_size=num_vectors,
            dimension=dim,
            insert_time=insert_time,
            insert_throughput=insert_throughput,
            index_build_time=insert_time,
            total_build_time=insert_time + index_create_time + meta_idx_time,
            index_size_mb=index_size_mb,
            retrieval_time_ms=retrieval_time_ms,
            retrieval_batch_time_ms=retrieval_batch_time_ms,
            search_qps=search_qps,
            search_latency_p50=latency_p50,
            search_latency_p95=latency_p95,
            search_latency_p99=latency_p99,
            recall_at_10=recall,
            params=params,
            filtered_search_qps=filtered_qps,
            filtered_search_latency_p50=filtered_p50,
            filtered_search_latency_p95=filtered_p95,
            filtered_recall_at_10=filtered_recall,
        )
        
        return result
        
    finally:
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
        
        # Create collection with ef_search=100
        collection = client.create_collection(
            name="sift1m",
            metadata={"hnsw:space": "l2", "hnsw:search_ef": 100}  # L2 distance, ef_search=100
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
                limit=k,
                search_params=SearchParams(ef=100)
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
                limit=k,
                search_params=SearchParams(ef=100)
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
                limit=k,
                search_params=SearchParams(ef=100)
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
                limit=k,
                search_params=SearchParams(ef=100)
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
        
        # Create collection with HNSW index and ef=100
        collection = client.collections.create(
            name=collection_name,
            properties=[
                Property(name="doc_id", data_type=DataType.INT),
                Property(name="category", data_type=DataType.INT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.L2_SQUARED,
                ef=100  # ef_search parameter
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


def save_results_json(results: List[BenchmarkResult], output_file: str, dataset_name: str = "SIFT1M"):
    """Save benchmark results to JSON file."""
    results_dict = {
        "benchmark": "Caliby vs ChromaDB",
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


# Default datasets directory
DEFAULT_DATASETS_DIR = '/home/zxjcarrot/Workspace/datasets'


def main():
    parser = argparse.ArgumentParser(description='Benchmark Caliby vs ChromaDB vs Qdrant vs Weaviate')
    parser.add_argument('--dataset', type=str, default='sift1m', 
                        choices=['sift1m', 'deep10m', 'arxiv', 'hnm'],
                        help='Dataset to use: sift1m (1M 128-dim), deep10m (10M 96-dim), arxiv (2.1M 384-dim), hnm (105K 2048-dim)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'Directory containing dataset (default: {DEFAULT_DATASETS_DIR}/<dataset>)')
    parser.add_argument('--num-vectors', type=int, default=None,
                        help='Limit number of vectors to use (default: all)')
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
    
    # Determine data directory based on dataset choice
    if args.data_dir is None:
        dataset_dirs = {
            'sift1m': os.path.join(DEFAULT_DATASETS_DIR, 'sift1m'),
            'deep10m': os.path.join(DEFAULT_DATASETS_DIR, 'deep10M'),
            'arxiv': os.path.join(DEFAULT_DATASETS_DIR, 'arxiv'),
            'hnm': os.path.join(DEFAULT_DATASETS_DIR, 'hnm'),
        }
        args.data_dir = dataset_dirs.get(args.dataset, os.path.join(DEFAULT_DATASETS_DIR, args.dataset))
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found")
        dataset_help = {
            'sift1m': "Please download SIFT1M dataset first",
            'deep10m': "Please download Deep10M dataset first",
            'arxiv': "Please download ArXiv dataset from https://storage.googleapis.com/ann-filtered-benchmark/arxiv.tar.gz",
            'hnm': "Please download H&M dataset from https://storage.googleapis.com/ann-filtered-benchmark/hnm.tgz"
        }
        print(dataset_help.get(args.dataset, f"Please download {args.dataset} dataset"))
        sys.exit(1)
    
    # Load dataset based on choice
    test_queries = None  # For filtered benchmark datasets
    payloads = None
    
    if args.dataset == 'sift1m':
        base_vectors, query_vectors, groundtruth = load_sift1m_data(args.data_dir)
        dataset_name = "SIFT1M"
    elif args.dataset == 'deep10m':
        base_vectors, query_vectors, groundtruth = load_deep10m_data(args.data_dir)
        dataset_name = "Deep10M"
    elif args.dataset == 'arxiv':
        base_vectors, test_queries, payloads = load_qdrant_filtered_dataset(args.data_dir, 'arxiv')
        query_vectors = np.array([q['query'] for q in test_queries], dtype=np.float32)
        # IMPORTANT: closest_ids in test queries are computed WITH filters applied!
        # We need to compute fresh unfiltered ground truth for unfiltered search benchmarks
        # This will be used for Weaviate/ChromaDB which do unfiltered search
        print("\nComputing unfiltered ground truth for ArXiv (cosine distance)...")
        groundtruth = compute_groundtruth(base_vectors, query_vectors, k=100, metric='cosine')
        dataset_name = "ArXiv"
    elif args.dataset == 'hnm':
        base_vectors, test_queries, payloads = load_qdrant_filtered_dataset(args.data_dir, 'hnm')
        query_vectors = np.array([q['query'] for q in test_queries], dtype=np.float32)
        # IMPORTANT: closest_ids in test queries are computed WITH filters applied!
        # We need to compute fresh unfiltered ground truth for unfiltered search benchmarks
        print("\nComputing unfiltered ground truth for H&M (cosine distance)...")
        groundtruth = compute_groundtruth(base_vectors, query_vectors, k=100, metric='cosine')
        dataset_name = "H&M"
    else:
        print(f"Error: Unknown dataset {args.dataset}")
        sys.exit(1)
    
    # Limit vectors if requested
    if args.num_vectors is not None:
        print(f"\nLimiting to {args.num_vectors} vectors for testing")
        base_vectors = base_vectors[:args.num_vectors]
        # IMPORTANT: Must recompute ground truth for the subset!
        # The original ground truth was computed against the full dataset,
        # so most IDs won't exist in our subset, leading to very low recall.
        # Use the correct distance metric for the dataset
        metric = 'cosine' if args.dataset in ['arxiv', 'hnm'] else 'l2'
        groundtruth = compute_groundtruth(base_vectors, query_vectors, k=100, metric=metric)
    
    # Benchmark parameters
    params = {
        'M': args.M,
        'ef_construction': args.ef_construction,
        'insert_batch_size': args.insert_batch_size,
        'test_queries': test_queries,  # For filtered benchmark datasets
        'payloads': payloads,  # Metadata for each vector
        'dataset_type': args.dataset,  # sift1m, deep10m, arxiv, hnm
    }
    
    results = []
    
    # Determine which benchmarks to skip based on --*-only flags
    only_flags = [args.caliby_only, args.chromadb_only, args.qdrant_only, args.weaviate_only]
    run_all = not any(only_flags)
    
    # Run Weaviate benchmark
    if (run_all or args.weaviate_only) and LIBS_AVAILABLE.get('weaviate', False):
        result = benchmark_weaviate(base_vectors, query_vectors, groundtruth, params)
        if result:
            results.append(result)
    elif args.weaviate_only:
        print("Weaviate not available, skipping...")
    
    # Run Caliby benchmark
    if (run_all or args.caliby_only) and LIBS_AVAILABLE.get('caliby', False):
        # Use filtered benchmark for arxiv/hnm datasets
        if args.dataset in ['arxiv', 'hnm']:
            result = benchmark_caliby_filtered(base_vectors, query_vectors, groundtruth, params)
        else:
            result = benchmark_caliby(base_vectors, query_vectors, groundtruth, params)
        if result:
            results.append(result)
    elif args.caliby_only:
        print("Caliby not available, skipping...")
    
    # Run ChromaDB benchmark
    if (run_all or args.chromadb_only) and LIBS_AVAILABLE.get('chromadb', False):
        result = benchmark_chromadb(base_vectors, query_vectors, groundtruth, params)
        if result:
            results.append(result)
    elif args.chromadb_only:
        print("ChromaDB not available, skipping...")
    
    # Run Qdrant benchmark (commented out)
    # if (run_all or args.qdrant_only) and LIBS_AVAILABLE.get('qdrant', False):
    #     result = benchmark_qdrant(base_vectors, query_vectors, groundtruth, params)
    #     if result:
    #         results.append(result)
    # elif args.qdrant_only:
    #     print("Qdrant not available, skipping...")
    
    
    # Print comparison
    if len(results) > 0:
        print_comparison_table(results)
        save_results_json(results, args.output, dataset_name)
    else:
        print("No benchmarks were run. Please install required libraries.")
        sys.exit(1)


if __name__ == '__main__':
    main()
