# Caliby Usage Guide

**High-Performance Vector Search with Efficient Larger-Than-Memory Support**

Caliby is a high-performance vector similarity search library that efficiently handles datasets larger than available memory. This guide covers all major features and APIs.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [System Configuration](#system-configuration)
4. [Vector Indexes](#vector-indexes)
   - [HNSW Index](#hnsw-index)
   - [DiskANN Index](#diskann-index)
   - [IVF+PQ Index](#ivfpq-index)
5. [Collections API](#collections-api)
   - [Schema Definition](#schema-definition)
   - [Document Operations](#document-operations)
   - [Vector Operations](#vector-operations)
   - [Text Search (BM25)](#text-search-bm25)
   - [Hybrid Search](#hybrid-search)
   - [Metadata Indexing](#metadata-indexing)
   - [Filtering](#filtering)
6. [Index Catalog](#index-catalog)
7. [Persistence & Recovery](#persistence--recovery)
8. [Performance Tips](#performance-tips)
9. [API Reference](#api-reference)

---

## Installation

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- Python 3.8+
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- CMake 3.15+
- OpenMP
- Abseil C++ library

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev libabsl-dev python3-dev
```

### Install from PyPI

```bash
pip install caliby
```

### Install from Source

```bash
git clone --recursive https://github.com/zxjcarrot/caliby.git
cd caliby
pip install -e .
```

---

## Quick Start

```python
import caliby
import numpy as np

# 1. Configure and initialize the system
caliby.set_buffer_config(size_gb=1.0)  # Set buffer pool size
caliby.open('/tmp/caliby_data')         # Open data directory

# 2. Create a collection with schema
schema = caliby.Schema()
schema.add_field("title", caliby.FieldType.STRING)
schema.add_field("category", caliby.FieldType.STRING)

collection = caliby.Collection("my_docs", schema, vector_dim=128)

# 3. Add documents with vectors
ids = [1, 2, 3]
contents = ["First document", "Second document", "Third document"]
metadatas = [
    {"title": "Doc 1", "category": "tech"},
    {"title": "Doc 2", "category": "science"},
    {"title": "Doc 3", "category": "tech"}
]
vectors = np.random.rand(3, 128).astype(np.float32).tolist()

collection.add(ids, contents, metadatas, vectors)

# 4. Create indexes
collection.create_hnsw_index("vec_idx")
collection.create_text_index("text_idx")

# 5. Search
query_vector = np.random.rand(128).astype(np.float32)
results = collection.search_vector(query_vector, "vec_idx", k=10)

for r in results:
    print(f"Doc {r.doc_id}: score={r.score:.4f}")

# 6. Close when done
caliby.close()
```

---

## System Configuration

### Buffer Pool Configuration

The buffer pool manages memory for all indexes. Configure it before opening any data.

```python
import caliby

# Set buffer pool size (in GB)
caliby.set_buffer_config(size_gb=2.0)

# Optional: set virtual memory limit
caliby.set_buffer_config(size_gb=2.0, virtgb=8.0)
```

### Initialize/Open Data Directory

```python
# Open data directory (creates if doesn't exist)
caliby.open('/path/to/data')

# Force cleanup existing data
caliby.open('/path/to/data', cleanup_if_exist=True)
```

### Shutdown

```python
# Flush and close all resources
caliby.close()

# Alternative: just flush without closing
caliby.flush_storage()
```

---

## Vector Indexes

Caliby provides three types of vector indexes for different use cases.

### HNSW Index

**Best for:** High recall requirements, moderate to large datasets, in-memory performance.

```python
import caliby
import numpy as np

# Initialize system
caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

# Create HNSW index
index = caliby.HnswIndex(
    max_elements=1_000_000,    # Maximum capacity
    dim=128,                    # Vector dimension
    M=16,                       # Connections per node (higher = better recall, more memory)
    ef_construction=200,        # Build-time search depth (higher = better quality, slower build)
    enable_prefetch=True,       # Enable memory prefetching
    skip_recovery=False,        # Set True to rebuild from scratch
    index_id=0,                 # Unique ID for multi-index support
    name='my_index'             # Optional human-readable name
)

# Add vectors (batch operation)
vectors = np.random.rand(10000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

# Single query search
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, ef_search=100)

print(f"Top 10 nearest neighbors: {labels}")
print(f"Distances: {distances}")

# Batch search (parallel)
queries = np.random.rand(100, 128).astype(np.float32)
labels, distances = index.search_knn_parallel(
    queries, k=10, ef_search=100, num_threads=4
)

# Get index info
print(f"Index name: {index.get_name()}")
print(f"Dimensions: {index.get_dim()}")
print(f"Was recovered: {index.was_recovered()}")

# Get statistics
stats = index.get_stats()
print(f"Distance computations: {stats['dist_comps']}")
print(f"Graph levels: {stats['num_levels']}")

# Flush to storage
index.flush()

caliby.close()
```

**Key Parameters:**
- `M`: Number of bi-directional links per node (default: 16). Higher values improve recall but use more memory.
- `ef_construction`: Size of dynamic candidate list during construction (default: 200). Higher values improve graph quality.
- `ef_search`: Search-time parameter controlling accuracy/speed tradeoff.

### DiskANN Index

**Best for:** Filtered search, dynamic updates, very large graphs with tags/labels.

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

# Create DiskANN index
index = caliby.DiskANN(
    dimensions=128,
    max_elements=1_000_000,
    R_max_degree=64,    # Max graph degree
    is_dynamic=True     # Enable dynamic inserts/deletes
)

# Prepare data with tags (for filtering)
vectors = np.random.rand(100000, 128).astype(np.float32)
tags = [[i % 100] for i in range(100000)]  # Each point has a tag 0-99

# Build parameters
params = caliby.BuildParams()
params.L_build = 100       # Build-time search depth
params.alpha = 1.2         # Pruning parameter
params.num_threads = 4

# Build the index
index.build(vectors, tags, params)

# Search parameters
search_params = caliby.SearchParams(L_search=50)
search_params.beam_width = 4

# Basic search
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search(query, K=10, params=search_params)

# Filtered search (only return vectors with tag=42)
labels, distances = index.search_with_filter(
    query, filter_label=42, K=10, params=search_params
)

# Parallel batch search
queries = np.random.rand(50, 128).astype(np.float32)
labels, distances = index.search_knn_parallel(
    queries, K=10, params=search_params, num_threads=4
)

# Dynamic operations (if is_dynamic=True)
new_point = np.random.rand(128).astype(np.float32)
index.insert_point(new_point, tags=[99], external_id=100000)
index.lazy_delete(external_id=100000)
# Note: consolidate_deletes is not yet fully implemented in the optimized version
# index.consolidate_deletes(params)

caliby.close()
```

### IVF+PQ Index

**Best for:** Very large datasets (10M+ vectors), memory-constrained environments.

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=0.5)
caliby.open('/tmp/caliby_data')

# Create IVF+PQ index
index = caliby.IVFPQIndex(
    max_elements=10_000_000,
    dim=128,
    num_clusters=256,           # Number of IVF clusters (K)
    num_subquantizers=8,        # PQ subquantizers (dim must be divisible by this)
    retrain_interval=10000,     # Retrain centroids periodically
    skip_recovery=False,
    index_id=0,
    name='large_dataset'
)

# IMPORTANT: Train the index first
training_data = np.random.rand(50000, 128).astype(np.float32)
index.train(training_data)

print(f"Index trained: {index.is_trained()}")

# Add points (after training)
vectors = np.random.rand(1000000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

# Search with nprobe parameter
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, nprobe=8)

# Parallel batch search
queries = np.random.rand(100, 128).astype(np.float32)
labels, distances = index.search_knn_parallel(
    queries, k=10, nprobe=16, num_threads=4
)

# Get statistics
stats = index.get_stats()
print(f"Clusters: {stats['num_clusters']}")
print(f"Avg list size: {stats['avg_list_size']}")

caliby.close()
```

---

## Collections API

Collections provide a document-oriented API with integrated vector search, text search, and metadata indexing.

### Schema Definition

```python
import caliby

schema = caliby.Schema()

# Add fields with different types
schema.add_field("title", caliby.FieldType.STRING)
schema.add_field("year", caliby.FieldType.INT)
schema.add_field("rating", caliby.FieldType.FLOAT)
schema.add_field("published", caliby.FieldType.BOOL)
schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
schema.add_field("scores", caliby.FieldType.INT_ARRAY)

# Optional: nullable parameter (default True)
schema.add_field("optional_field", caliby.FieldType.STRING, nullable=True)

# Check fields
print(schema.has_field("title"))  # True
print(schema.fields())            # List of FieldDef objects
```

**Field Types:**
- `STRING` - Text strings
- `INT` - Integers
- `FLOAT` - Floating-point numbers
- `BOOL` - Boolean values
- `STRING_ARRAY` - Array of strings
- `INT_ARRAY` - Array of integers

### Document Operations

```python
import caliby

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

schema = caliby.Schema()
schema.add_field("title", caliby.FieldType.STRING)
schema.add_field("category", caliby.FieldType.STRING)
schema.add_field("year", caliby.FieldType.INT)

# Create collection (without vectors)
collection = caliby.Collection("docs", schema)

# Or with vector support
collection_with_vec = caliby.Collection(
    "vec_docs", schema, 
    vector_dim=128,
    distance_metric=caliby.DistanceMetric.COSINE  # L2, COSINE, or IP
)

# Add documents (batch)
ids = [1, 2, 3, 4, 5]
contents = ["Doc one", "Doc two", "Doc three", "Doc four", "Doc five"]
metadatas = [
    {"title": "First", "category": "A", "year": 2020},
    {"title": "Second", "category": "B", "year": 2021},
    {"title": "Third", "category": "A", "year": 2022},
    {"title": "Fourth", "category": "B", "year": 2023},
    {"title": "Fifth", "category": "A", "year": 2024}
]

collection.add(ids, contents, metadatas)

print(f"Document count: {collection.doc_count()}")

# Get documents by ID
docs = collection.get([1, 3, 5])
for doc in docs:
    print(f"ID: {doc.id}, Content: {doc.content}, Meta: {doc.metadata}")

# Update metadata
collection.update([1, 2], [
    {"title": "Updated First", "category": "A", "year": 2025},
    {"title": "Updated Second", "category": "C", "year": 2025}
])

# Delete documents
collection.delete([4, 5])

# Open existing collection
existing = caliby.Collection.open("docs")

caliby.close()
```

### Vector Operations

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

schema = caliby.Schema()
schema.add_field("name", caliby.FieldType.STRING)

# Create collection with vectors
collection = caliby.Collection("vectors", schema, vector_dim=64)

# Add documents with vectors
ids = [1, 2, 3]
contents = ["First", "Second", "Third"]
metadatas = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
vectors = np.random.rand(3, 64).astype(np.float32).tolist()

collection.add(ids, contents, metadatas, vectors)

# Add vectors to existing documents
more_ids = [4, 5]
more_contents = ["Fourth", "Fifth"]
more_metadatas = [{"name": "D"}, {"name": "E"}]
collection.add(more_ids, more_contents, more_metadatas)

# Add vectors separately
more_vectors = np.random.rand(2, 64).astype(np.float32)
collection.add_vectors(more_ids, more_vectors)

# Create HNSW index for vector search
collection.create_hnsw_index("vec_idx", M=16, ef_construction=200)

# Or DiskANN index
collection.create_diskann_index("disk_idx", R=64, L=100, alpha=1.2)

# Vector search
query = np.random.rand(64).astype(np.float32)
results = collection.search_vector(query, "vec_idx", k=5)

for r in results:
    print(f"Doc {r.doc_id}: score={r.score:.4f}")

caliby.close()
```

### Text Search (BM25)

```python
import caliby

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

schema = caliby.Schema()
schema.add_field("title", caliby.FieldType.STRING)

collection = caliby.Collection("articles", schema)

# Add documents
ids = [1, 2, 3, 4]
contents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for pattern recognition",
    "Natural language processing enables text understanding",
    "Computer vision helps machines interpret images"
]
metadatas = [
    {"title": "ML Intro"},
    {"title": "Deep Learning"},
    {"title": "NLP Guide"},
    {"title": "CV Tutorial"}
]

collection.add(ids, contents, metadatas)

# Create text index for BM25 search
collection.create_text_index("text_idx")

# Text search
results = collection.search_text("machine learning neural", "text_idx", k=3)

for r in results:
    print(f"Doc {r.doc_id}: text_score={r.text_score:.4f}")
    if r.document:
        print(f"  Content: {r.document['content'][:50]}...")

caliby.close()
```

### Hybrid Search

Combine vector and text search with score fusion.

```python
import caliby
import numpy as np

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

schema = caliby.Schema()
schema.add_field("title", caliby.FieldType.STRING)

collection = caliby.Collection("hybrid_docs", schema, vector_dim=128)

# Add documents with vectors
ids = [1, 2, 3]
contents = [
    "Introduction to machine learning algorithms",
    "Deep neural network architectures",
    "Statistical methods in data science"
]
metadatas = [{"title": f"Doc {i}"} for i in ids]
vectors = np.random.rand(3, 128).astype(np.float32).tolist()

collection.add(ids, contents, metadatas, vectors)

# Create both indexes
collection.create_hnsw_index("vec_idx")
collection.create_text_index("text_idx")

# Configure fusion parameters
fusion = caliby.FusionParams()
fusion.method = caliby.FusionMethod.RRF      # Reciprocal Rank Fusion
fusion.rrf_k = 60                            # RRF constant

# Or use weighted fusion
# fusion.method = caliby.FusionMethod.WEIGHTED
# fusion.vector_weight = 0.7
# fusion.text_weight = 0.3

# Hybrid search
query_vec = np.random.rand(128).astype(np.float32)
query_text = "machine learning"

results = collection.search_hybrid(
    query_vec, "vec_idx",
    query_text, "text_idx",
    k=5,
    fusion=fusion
)

for r in results:
    print(f"Doc {r.doc_id}: combined={r.score:.4f}, "
          f"vector={r.vector_score:.4f}, text={r.text_score:.4f}")

caliby.close()
```

### Metadata Indexing

Create indexes on metadata fields for efficient filtering and queries.

```python
import caliby

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

schema = caliby.Schema()
schema.add_field("category", caliby.FieldType.STRING)
schema.add_field("year", caliby.FieldType.INT)
schema.add_field("price", caliby.FieldType.FLOAT)

collection = caliby.Collection("products", schema)

# Single-field index
collection.create_metadata_index("year_idx", ["year"])

# Composite index (leftmost prefix rule applies)
# Can efficiently query: category, or category+year, or category+year+price
# Cannot efficiently query: year alone, or year+price
collection.create_metadata_index("category_year_idx", ["category", "year"])

# Unique index (enforces uniqueness on the full composite key)
collection.create_metadata_index(
    "unique_idx", ["category", "year"], unique=True
)

# Legacy API (single field only)
collection.create_btree_index("price_btree", "price")

# List all indexes
indexes = collection.list_indices()
for idx in indexes:
    print(f"Index: {idx['name']}, Type: {idx['type']}, Config: {idx['config']}")

# Drop an index
collection.drop_index("price_btree")

caliby.close()
```

### Filtering

Apply filters to search operations using a JSON DSL.

```python
import caliby
import numpy as np
import json

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

schema = caliby.Schema()
schema.add_field("category", caliby.FieldType.STRING)
schema.add_field("year", caliby.FieldType.INT)
schema.add_field("price", caliby.FieldType.FLOAT)
schema.add_field("active", caliby.FieldType.BOOL)

collection = caliby.Collection("filtered", schema, vector_dim=64)

# Add sample data
ids = list(range(1, 101))
contents = [f"Product {i}" for i in ids]
metadatas = [
    {
        "category": ["tech", "home", "office"][i % 3],
        "year": 2020 + (i % 5),
        "price": 10.0 + (i * 0.5),
        "active": i % 2 == 0
    }
    for i in ids
]
vectors = np.random.rand(100, 64).astype(np.float32).tolist()

collection.add(ids, contents, metadatas, vectors)
collection.create_hnsw_index("vec_idx")
collection.create_text_index("text_idx")

# Simple equality filter
filter1 = json.dumps({"field": "category", "op": "eq", "value": "tech"})

# Comparison filter
filter2 = json.dumps({"field": "year", "op": "gte", "value": 2023})

# AND condition
filter3 = json.dumps({
    "and": [
        {"field": "category", "op": "eq", "value": "tech"},
        {"field": "price", "op": "lt", "value": 50.0}
    ]
})

# OR condition
filter4 = json.dumps({
    "or": [
        {"field": "category", "op": "eq", "value": "tech"},
        {"field": "category", "op": "eq", "value": "home"}
    ]
})

# NOT condition
filter5 = json.dumps({
    "not": {"field": "active", "op": "eq", "value": False}
})

# IN operator
filter6 = json.dumps({
    "field": "year", "op": "in", "value": [2022, 2023, 2024]
})

# Apply filter to vector search
query = np.random.rand(64).astype(np.float32)
results = collection.search_vector(query, "vec_idx", k=10, filter=filter3)

# Apply filter to text search
results = collection.search_text("product", "text_idx", k=10, filter=filter1)

# Apply filter to hybrid search
fusion = caliby.FusionParams()
results = collection.search_hybrid(
    query, "vec_idx",
    "product", "text_idx",
    k=10, fusion=fusion, filter=filter4
)

caliby.close()
```

**Filter Operators:**
- `eq` - Equal
- `ne` - Not equal
- `gt` - Greater than
- `gte` - Greater than or equal
- `lt` - Less than
- `lte` - Less than or equal
- `in` - Value in list
- `contains` - String contains substring

---

## Index Catalog

The Index Catalog provides centralized management for standalone indexes.

```python
import caliby

caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/caliby_data')

# Get the singleton catalog instance
catalog = caliby.IndexCatalog.instance()

print(f"Catalog initialized: {catalog.is_initialized()}")
print(f"Data directory: {catalog.data_dir()}")

# Create indexes with simplified API
hnsw_handle = catalog.create_hnsw_index(
    name="embeddings",
    dimensions=128,
    max_elements=100000,
    M=16,
    ef_construction=200
)

diskann_handle = catalog.create_diskann_index(
    name="vectors",
    dimensions=256,
    max_elements=50000,
    R_max_degree=64,
    L_build=100,
    alpha=1.2
)

# List all indexes
indexes = catalog.list_indexes()
for info in indexes:
    print(f"Index: {info.name}, Type: {info.type}, Elements: {info.num_elements}")

# Check if index exists
if catalog.index_exists("embeddings"):
    print("embeddings index exists")

# Get detailed info
info = catalog.get_index_info("embeddings")
print(f"Created: {info.create_time}, File: {info.file_path}")

# Open existing index
handle = catalog.open_index("embeddings")

# Flush changes
handle.flush()

# Drop index
catalog.drop_index("vectors")

caliby.close()
```

---

## Persistence & Recovery

Caliby automatically persists data through the buffer pool.

```python
import caliby
import numpy as np

# First session: create and populate
caliby.set_buffer_config(size_gb=1.0)
caliby.open('/path/to/persistent_data')

schema = caliby.Schema()
schema.add_field("name", caliby.FieldType.STRING)

collection = caliby.Collection("persistent", schema, vector_dim=64)
collection.add([1, 2, 3], ["A", "B", "C"], 
               [{"name": n} for n in ["a", "b", "c"]],
               np.random.rand(3, 64).astype(np.float32).tolist())
collection.create_hnsw_index("vec_idx")

# Explicit flush (optional - happens automatically on close)
collection.flush()

caliby.close()

# Second session: recovery
# Note: buffer_config persists in same process; only call before first open()
caliby.open('/path/to/persistent_data')

# Open existing collection
recovered = caliby.Collection.open("persistent")
print(f"Recovered {recovered.doc_count()} documents")

# HNSW index also supports recovery
index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=64,
    M=16,
    ef_construction=200,
    enable_prefetch=True,
    skip_recovery=False,  # Set False to enable recovery
    index_id=1,
    name='my_index'
)

if index.was_recovered():
    print("Index recovered from disk!")

caliby.close()
```

---

## Performance Tips

### Buffer Pool Sizing

- Set buffer pool size based on your working set
- For in-memory workloads: buffer size ≥ dataset size
- For larger-than-memory: start with 20-50% of dataset size

```python
# For a 10GB dataset
caliby.set_buffer_config(size_gb=4.0)  # 40% of dataset
```

### HNSW Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `M` | Graph connectivity | 16-64 (higher = better recall, more memory) |
| `ef_construction` | Build quality | 100-500 (higher = better graph, slower build) |
| `ef_search` | Search accuracy | Start with 50-100, increase for better recall |

```python
# High recall configuration
index = caliby.HnswIndex(M=32, ef_construction=400, ...)
results = index.search_knn(query, k=10, ef_search=200)

# Fast search configuration
index = caliby.HnswIndex(M=16, ef_construction=200, ...)
results = index.search_knn(query, k=10, ef_search=50)
```

### IVF+PQ Tuning

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `num_clusters` | Search granularity | √n to 4√n where n is dataset size |
| `num_subquantizers` | Compression ratio | 8-32 (must divide dimension evenly) |
| `nprobe` | Search accuracy | 1-32 (higher = more accurate, slower) |

```python
# For 1M vectors
index = caliby.IVFPQIndex(
    max_elements=1_000_000,
    dim=128,
    num_clusters=1000,   # ~√1M
    num_subquantizers=16
)
```

### Parallel Operations

```python
# Parallel insertion
index.add_points(vectors, num_threads=8)

# Parallel search
results = index.search_knn_parallel(queries, k=10, ef_search=100, num_threads=8)
```

### Force Eviction

For memory-constrained scenarios:

```python
# Evict a portion of buffer pool (0.0 to 1.0)
caliby.force_evict_buffer_portion(0.5)  # Evict 50%
```

---

## API Reference

### Global Functions

| Function | Description |
|----------|-------------|
| `caliby.set_buffer_config(size_gb, virtgb=None)` | Configure buffer pool |
| `caliby.open(path, cleanup_if_exist=False)` | Open data directory |
| `caliby.close()` | Close and flush all resources |
| `caliby.flush_storage()` | Flush without closing |
| `caliby.force_evict_buffer_portion(portion)` | Evict buffer pages |

### HnswIndex

| Method | Description |
|--------|-------------|
| `add_points(vectors, num_threads=0)` | Add vectors in batch |
| `search_knn(query, k, ef_search)` | Single query search |
| `search_knn_parallel(queries, k, ef_search, num_threads=0)` | Batch search |
| `flush()` | Flush to storage |
| `get_name()` | Get index name |
| `get_dim()` | Get vector dimension |
| `was_recovered()` | Check if recovered from disk |
| `get_stats()` | Get statistics dict |
| `reset_stats()` | Reset statistics |

### DiskANN

| Method | Description |
|--------|-------------|
| `build(data, tags, params)` | Build index |
| `search(query, K, params)` | Basic search |
| `search_with_filter(query, filter_label, K, params)` | Filtered search |
| `search_knn_parallel(queries, K, params, num_threads=0)` | Batch search |
| `insert_point(point, tags, external_id)` | Insert single point |
| `lazy_delete(external_id)` | Mark for deletion |
| `consolidate_deletes(params)` | Repair graph |

### IVFPQIndex

| Method | Description |
|--------|-------------|
| `train(training_data)` | Train centroids/codebooks |
| `add_points(items, num_threads=0)` | Add vectors |
| `search_knn(query, k, nprobe)` | Single query search |
| `search_knn_parallel(queries, k, nprobe, num_threads=0)` | Batch search |
| `is_trained()` | Check if trained |
| `get_count()` | Get vector count |
| `get_stats()` | Get statistics |

### Collection

| Method | Description |
|--------|-------------|
| `add(ids, contents, metadatas, vectors=[])` | Add documents |
| `add_vectors(ids, vectors)` | Add vectors to existing docs |
| `get(ids)` | Get documents by ID |
| `update(ids, metadatas)` | Update metadata |
| `delete(ids)` | Delete documents |
| `create_hnsw_index(name, M=16, ef_construction=200)` | Create HNSW index |
| `create_diskann_index(name, R=64, L=100, alpha=1.2)` | Create DiskANN index |
| `create_text_index(name)` | Create BM25 text index |
| `create_metadata_index(name, fields, unique=False)` | Create metadata index |
| `search_vector(query, index_name, k, filter="")` | Vector search |
| `search_text(query, index_name, k, filter="")` | Text search |
| `search_hybrid(query_vec, vec_idx, query_text, text_idx, k, fusion, filter="")` | Hybrid search |
| `list_indices()` | List all indexes |
| `drop_index(name)` | Drop an index |
| `flush()` | Flush to storage |

### Enums

```python
# Distance metrics
caliby.DistanceMetric.L2      # Euclidean distance
caliby.DistanceMetric.COSINE  # Cosine similarity
caliby.DistanceMetric.IP      # Inner product

# Field types
caliby.FieldType.STRING
caliby.FieldType.INT
caliby.FieldType.FLOAT
caliby.FieldType.BOOL
caliby.FieldType.STRING_ARRAY
caliby.FieldType.INT_ARRAY

# Fusion methods
caliby.FusionMethod.RRF       # Reciprocal Rank Fusion
caliby.FusionMethod.WEIGHTED  # Weighted combination

# Index types
caliby.IndexType.HNSW
caliby.IndexType.DISKANN
caliby.IndexType.IVF
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/zxjcarrot/caliby/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zxjcarrot/caliby/discussions)
- **Email**: xinjing@mit.edu
