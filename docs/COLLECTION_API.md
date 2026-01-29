# Caliby Collection API

Caliby provides a high-level Collection API for managing documents with vectors, text content, and metadata. This guide covers the complete workflow from creating collections to performing filtered vector searches.

## Table of Contents

- [Quick Start](#quick-start)
- [Collections](#collections)
- [Schema Definition](#schema-definition)
- [Adding Documents](#adding-documents)
- [Creating Indices](#creating-indices)
- [Vector Search](#vector-search)
- [Filtered Vector Search](#filtered-vector-search)
- [Hybrid Search](#hybrid-search)
- [Best Practices](#best-practices)

## Quick Start

```python
import caliby
import numpy as np

# Initialize Caliby
caliby.open("/path/to/data")

# Create a schema
schema = caliby.Schema()
schema.add_field("category", caliby.FieldType.INT)
schema.add_field("title", caliby.FieldType.STRING)

# Create a collection with 128-dimensional vectors
collection = caliby.Collection("my_collection", schema, vector_dim=128)

# Create HNSW index for vector search
collection.create_hnsw_index("vec_idx", M=16, ef_construction=200)

# Create metadata index for filtered search
collection.create_metadata_index("category_idx", field="category")

# Add documents
vectors = np.random.rand(1000, 128).astype(np.float32)
contents = ["Document " + str(i) for i in range(1000)]
metadatas = [{"category": i % 10, "title": f"Title {i}"} for i in range(1000)]

doc_ids = collection.add(contents, metadatas, vectors.tolist())

# Vector search
results = collection.search_vector(query_vector, "vec_idx", k=10)

# Filtered vector search
results = collection.search_vector(
    query_vector, "vec_idx", k=10,
    filter={"category": 5}  # Only return documents where category == 5
)

# Cleanup
caliby.close()
```

## Collections

### Creating a Collection

```python
import caliby

# Initialize the system first
caliby.open("/path/to/data/directory")

# Create schema for collection metadata
schema = caliby.Schema()
schema.add_field("category", caliby.FieldType.INT)
schema.add_field("author", caliby.FieldType.STRING)
schema.add_field("price", caliby.FieldType.FLOAT)
schema.add_field("published", caliby.FieldType.BOOL)

# Create collection
# - name: unique identifier for the collection
# - schema: defines metadata fields
# - vector_dim: dimension of vectors (0 if no vectors)
# - distance_metric: L2 (default), IP (inner product), or COSINE
collection = caliby.Collection(
    name="products",
    schema=schema,
    vector_dim=384,
    distance_metric=caliby.DistanceMetric.L2
)
```

### Opening an Existing Collection

```python
# Re-open a previously created collection
collection = caliby.Collection.open("products")
```

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `L2` | Euclidean distance (default) | General purpose, normalized vectors |
| `IP` | Inner product (negative) | Maximum inner product search |
| `COSINE` | Cosine similarity | Text embeddings, when magnitude doesn't matter |

## Schema Definition

Schemas define the structure of document metadata.

### Field Types

```python
schema = caliby.Schema()

# Basic types
schema.add_field("name", caliby.FieldType.STRING)
schema.add_field("count", caliby.FieldType.INT)
schema.add_field("score", caliby.FieldType.FLOAT)
schema.add_field("active", caliby.FieldType.BOOL)

# Array types
schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
schema.add_field("ids", caliby.FieldType.INT_ARRAY)

# Nullable fields (default is nullable=True)
schema.add_field("optional_field", caliby.FieldType.STRING, nullable=True)
schema.add_field("required_field", caliby.FieldType.STRING, nullable=False)
```

## Adding Documents

### Basic Addition

```python
# Add documents with content, metadata, and vectors
contents = ["First document", "Second document", "Third document"]
metadatas = [
    {"category": 1, "author": "Alice"},
    {"category": 2, "author": "Bob"},
    {"category": 1, "author": "Charlie"}
]
vectors = [
    [0.1, 0.2, ...],  # 128-dim vector
    [0.3, 0.4, ...],
    [0.5, 0.6, ...]
]

# Returns list of assigned document IDs
doc_ids = collection.add(contents, metadatas, vectors)
print(f"Added documents: {doc_ids}")  # [1, 2, 3]
```

### Batch Addition (Recommended for Large Datasets)

```python
import numpy as np

# For best performance, add documents in batches
batch_size = 10000
total_docs = 1000000

for i in range(0, total_docs, batch_size):
    batch_end = min(i + batch_size, total_docs)
    
    batch_contents = [f"Document {j}" for j in range(i, batch_end)]
    batch_metadatas = [{"category": j % 10} for j in range(i, batch_end)]
    batch_vectors = vectors[i:batch_end].tolist()
    
    collection.add(batch_contents, batch_metadatas, batch_vectors)
    print(f"Inserted {batch_end}/{total_docs} documents...")
```

## Creating Indices

### HNSW Index (Vector Search)

HNSW (Hierarchical Navigable Small World) provides fast approximate nearest neighbor search.

```python
# Create HNSW index
collection.create_hnsw_index(
    name="vec_idx",
    M=16,              # Number of connections per layer (default: 16)
    ef_construction=200 # Construction time quality (default: 200)
)
```

**Parameter Guidelines:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `M` | 8-64 | Higher = better recall, more memory |
| `ef_construction` | 50-500 | Higher = better index quality, slower build |

### Text Index (BM25 Search)

```python
# Create text index for full-text search
collection.create_text_index(
    name="text_idx",
    analyzer="standard",  # "standard", "whitespace", or "none"
    language="english",   # For stemming
    k1=1.2,              # BM25 term frequency saturation
    b=0.75               # BM25 document length normalization
)
```

### Metadata Index (Filtered Search)

**Important:** Create metadata indices on fields you'll use for filtering to enable optimized filtered search.

```python
# Single-field index
collection.create_metadata_index(
    name="category_idx",
    field="category"
)

# Composite index (multiple fields)
collection.create_metadata_index(
    name="composite_idx",
    fields=["category", "author"]
)
```

## Vector Search

### Basic Vector Search

```python
# Search for k nearest neighbors
query_vector = [0.1, 0.2, ...]  # Same dimension as indexed vectors

results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=10
)

# Process results
for result in results:
    print(f"Doc ID: {result.doc_id}")
    print(f"Distance: {result.score}")
    print(f"Content: {result.document.content}")
    print(f"Metadata: {result.document.metadata}")
```

### Search Parameters

```python
# Control search quality with ef_search
results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=10,
    params={"ef_search": 200}  # Higher = better recall, slower search
)
```

## Filtered Vector Search

Filtered search finds the k nearest neighbors among documents matching a filter condition.

### Filter Syntax

Caliby uses MongoDB-style filter syntax:

```python
# Equality filter
filter = {"category": 5}

# Comparison operators
filter = {"price": {"$gt": 100}}      # Greater than
filter = {"price": {"$gte": 100}}     # Greater than or equal
filter = {"price": {"$lt": 50}}       # Less than
filter = {"price": {"$lte": 50}}      # Less than or equal
filter = {"price": {"$ne": 0}}        # Not equal

# Set membership
filter = {"category": {"$in": [1, 2, 3]}}     # In set
filter = {"category": {"$nin": [4, 5, 6]}}    # Not in set

# Array contains
filter = {"tags": {"$contains": "featured"}}

# Logical operators
filter = {"$and": [{"category": 1}, {"price": {"$lt": 100}}]}
filter = {"$or": [{"category": 1}, {"category": 2}]}
```

### Filtered Search Examples

```python
import json

# Simple equality filter
results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=10,
    filter=json.dumps({"category": 5})
)

# Range filter
results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=10,
    filter=json.dumps({"price": {"$gte": 10, "$lte": 100}})
)

# Combined filters
results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=10,
    filter=json.dumps({
        "$and": [
            {"category": {"$in": [1, 2, 3]}},
            {"active": True}
        ]
    })
)
```

### How Filtered Search Works

Caliby uses **post-search filtering with over-fetching**:

1. Search for `k * multiplier` nearest neighbors in HNSW
2. Filter results by the condition
3. If not enough results, retry with larger ef_search parameters 

### Performance Tips for Filtered Search

```python
# 1. Create metadata index on filter fields
collection.create_metadata_index("category_idx", field="category")

# 2. For very selective filters (< 1% of data), increase ef_search
results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=10,
    filter=json.dumps({"rare_category": 999}),
    params={"ef_search": 500}
)

# 3. For high recall requirements, search for more results
results = collection.search_vector(
    vector=query_vector,
    index_name="vec_idx",
    k=50,  # Search for more, then take top 10
    filter=json.dumps({"category": 5})
)[:10]
```

## Hybrid Search

Combine vector similarity with text relevance using fusion.

```python
# Hybrid search with RRF (Reciprocal Rank Fusion)
results = collection.search_hybrid(
    vector=query_vector,
    vector_index_name="vec_idx",
    text="search query terms",
    text_index_name="text_idx",
    k=10,
    fusion={
        "method": "rrf",  # or "weighted"
        "rrf_k": 60       # RRF constant
    }
)

# Hybrid search with weighted fusion
results = collection.search_hybrid(
    vector=query_vector,
    vector_index_name="vec_idx",
    text="search query",
    text_index_name="text_idx",
    k=10,
    fusion={
        "method": "weighted",
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "normalize": True
    }
)
```

## Best Practices

### 1. Index Creation Order

```python
# Always create indices BEFORE adding documents
collection = caliby.Collection("my_collection", schema, vector_dim=128)

# Create indices first
collection.create_hnsw_index("vec_idx", M=16, ef_construction=200)
collection.create_metadata_index("category_idx", field="category")
collection.create_text_index("text_idx")

# Then add documents
collection.add(contents, metadatas, vectors)
```

### 2. Batch Size for Insertion

```python
# Recommended batch sizes based on vector dimension
# - 128-dim: 10,000 per batch
# - 384-dim: 5,000 per batch
# - 768-dim: 2,500 per batch
# - 1536-dim: 1,000 per batch
```

### 3. Memory Management

```python
# Flush to persist changes to disk
collection.flush()

# Close Caliby when done
caliby.close()
```

## Performance Benchmarks

On SIFT1M dataset (1M 128-dim vectors):

| Operation | Caliby | Weaviate | ChromaDB |
|-----------|--------|----------|----------|
| Insert (docs/s) | 21,000 | 3,000 | 4,300 |
| Vector Search QPS | 5,900 | 300 | 1,900 |
| Filtered Search QPS | 1,300 | 105 | N/A |
| Filtered Recall@10 | 99.5% | 99.9% | N/A |
| Storage Size | 1.1 GB | 2.2 GB | 1.4 GB |

## Error Handling

```python
try:
    results = collection.search_vector(query_vector, "vec_idx", k=10)
except RuntimeError as e:
    if "not found" in str(e):
        print("Index does not exist")
    elif "not initialized" in str(e):
        print("Call caliby.open() first")
    else:
        raise
```

## Complete Example

```python
import caliby
import numpy as np
import json

# Initialize
caliby.open("/tmp/caliby_example")

# Define schema
schema = caliby.Schema()
schema.add_field("category", caliby.FieldType.INT)
schema.add_field("price", caliby.FieldType.FLOAT)
schema.add_field("tags", caliby.FieldType.STRING_ARRAY)

# Create collection
collection = caliby.Collection(
    name="products",
    schema=schema,
    vector_dim=128,
    distance_metric=caliby.DistanceMetric.L2
)

# Create indices
collection.create_hnsw_index("vec_idx", M=16, ef_construction=200)
collection.create_metadata_index("category_idx", field="category")
collection.create_text_index("text_idx")

# Add sample data
np.random.seed(42)
n_docs = 10000

contents = [f"Product {i}: A great item in category {i % 10}" for i in range(n_docs)]
metadatas = [
    {
        "category": i % 10,
        "price": float(10 + (i % 100)),
        "tags": ["featured"] if i % 5 == 0 else ["regular"]
    }
    for i in range(n_docs)
]
vectors = np.random.rand(n_docs, 128).astype(np.float32).tolist()

doc_ids = collection.add(contents, metadatas, vectors)
print(f"Added {len(doc_ids)} documents")

# Vector search
query = np.random.rand(128).astype(np.float32).tolist()
results = collection.search_vector(query, "vec_idx", k=5)
print("\nVector search results:")
for r in results:
    print(f"  Doc {r.doc_id}: distance={r.score:.4f}")

# Filtered search - only category 3
results = collection.search_vector(
    query, "vec_idx", k=5,
    filter=json.dumps({"category": 3})
)
print("\nFiltered search (category=3):")
for r in results:
    print(f"  Doc {r.doc_id}: distance={r.score:.4f}, category={r.document.metadata['category']}")

# Filtered search with price range
results = collection.search_vector(
    query, "vec_idx", k=5,
    filter=json.dumps({
        "$and": [
            {"category": {"$in": [1, 2, 3]}},
            {"price": {"$lte": 50}}
        ]
    })
)
print("\nFiltered search (category in [1,2,3] AND price <= 50):")
for r in results:
    meta = r.document.metadata
    print(f"  Doc {r.doc_id}: distance={r.score:.4f}, category={meta['category']}, price={meta['price']}")

# Cleanup
collection.flush()
caliby.close()
print("\nDone!")
```
