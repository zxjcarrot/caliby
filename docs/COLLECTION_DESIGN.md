# Caliby Collection System Design

## Overview

A **Collection** is a typed document store with optional vector search capabilities. Collections support:
- Structured metadata with schema enforcement
- Multiple attachable indices (vector, text, B-tree)
- Hybrid search combining vector similarity and text relevance
- Adaptive filtered search based on selectivity

---

## 1. Core Concepts

### 1.1 Collection

A collection stores documents with typed metadata. Vector embeddings are optional.

```python
import caliby

caliby.set_buffer_config(10.0)
# Text-only collection (no vector search)
articles = caliby.Collection(
    name="articles",
    schema={
        "title": "string",
        "author": "string", 
        "year": "int",
        "tags": "string[]",
        "price": "float"
    }
)

# Collection with vector support
embeddings_collection = caliby.Collection(
    name="embeddings",
    schema={
        "title": "string",
        "category": "string"
    },
    vector_dim=768,           # enables vector indices
    distance_metric="cosine"  # "cosine" | "l2" | "ip"
)
```

### 1.2 Schema Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | UTF-8 text | `"hello"` |
| `int` | 64-bit signed integer | `42` |
| `float` | 64-bit float | `3.14` |
| `bool` | Boolean | `true` |
| `string[]` | Array of strings | `["a", "b"]` |
| `int[]` | Array of integers | `[1, 2, 3]` |

### 1.3 Documents

Documents consist of:
- **id**: Unique integer identifier (user-provided or auto-generated)
- **content**: Optional text content (for text search, stored with document)
- **metadata**: Typed key-value pairs (schema-enforced, stored with document)
- **vector**: Optional embedding (if collection has `vector_dim`, stored in vector index)

```python
# Add documents (text-only collection)
articles.add(
    ids=[1, 2],  # integer IDs
    contents=[
        "The quick brown fox jumps over the lazy dog",
        "Machine learning enables computers to learn from data"
    ],
    metadatas=[
        {"title": "Fox Story", "author": "alice", "year": 2024, "tags": ["nature"], "price": 9.99},
        {"title": "ML Intro", "author": "bob", "year": 2023, "tags": ["tech", "ai"], "price": 29.99}
    ]
)

# Add documents with vectors
embeddings_collection.add(
    ids=[100, 101],  # integer IDs
    contents=["Document one", "Document two"],
    metadatas=[{"title": "Doc 1", "category": "A"}, {"title": "Doc 2", "category": "B"}],
    vectors=np.array([[...], [...]], dtype=np.float32)  # shape: (2, 768)
)

# Add vectors later (for existing documents)
embeddings_collection.add_vectors(
    ids=[100, 101],  # reference by integer ID
    vectors=new_embeddings
)
```

---

## 2. Index System

### 2.1 Index Types

| Type | Purpose | File Extension |
|------|---------|----------------|
| `hnsw` | In-memory graph ANN | `.hnsw` |
| `diskann` | Disk-based graph ANN | `.diskann` |
| `ivfpq` | Quantized inverted file | `.ivfpq` |
| `text` | BM25 inverted index | `.text` |
| `btree` | Ordered metadata index | `.btree` |

### 2.2 Creating Indices

Indices are created explicitly and tracked by the catalog:

```python
# Vector index (requires collection with vector_dim)
articles.create_index(
    name="vec_hnsw",
    type="hnsw",
    config={"M": 16, "ef_construction": 100}
)

# Alternative vector index
articles.create_index(
    name="vec_diskann", 
    type="diskann",
    config={"R": 32, "L": 100, "alpha": 1.2}
)

# Text index on content field
articles.create_index(
    name="content_search",
    type="text",
    config={
        "fields": ["content"],      # which fields to index
        "analyzer": "standard",     # "standard" | "whitespace" | "none"
        "language": "english"       # for stemming/stopwords
    }
)

# B-tree index on metadata field (for range queries)
articles.create_index(
    name="year_idx",
    type="btree",
    config={"field": "year"}
)

# List indices
articles.list_indices()
# [
#   {"name": "vec_hnsw", "type": "hnsw", "status": "ready"},
#   {"name": "content_search", "type": "text", "status": "ready"},
#   {"name": "year_idx", "type": "btree", "status": "ready"}
# ]

# Drop index
articles.drop_index("vec_diskann")
```

### 2.3 Catalog Registration

All indices are registered in the catalog with:
- Unique index ID
- Parent collection ID
- Index type and configuration
- File path for storage
- Status (building, ready, error)

```
catalog/
├── catalog.meta          # Global catalog metadata
├── collections/
│   ├── articles.col      # Collection metadata + document schema
│   └── embeddings.col
└── indices/
    ├── articles.vec_hnsw.hnsw
    ├── articles.content_search.text
    ├── articles.year_idx.btree
    └── embeddings.vec_diskann.diskann
```

---

## 3. Query API

### 3.1 Vector Search

```python
# Basic vector search
results = collection.search(
    vector=query_embedding,
    index="vec_hnsw",
    k=10
)

# With search parameters
results = collection.search(
    vector=query_embedding,
    index="vec_hnsw",
    k=10,
    params={"ef_search": 100}  # index-specific params
)
```

### 3.2 Text Search

```python
# BM25 text search
results = collection.search(
    text="machine learning tutorial",
    index="content_search",
    k=10
)
```

### 3.3 Filtered Search

```python
# Vector search with metadata filter
results = collection.search(
    vector=query_embedding,
    index="vec_hnsw",
    k=10,
    where={
        "$and": [
            {"year": {"$gte": 2023}},
            {"tags": {"$contains": "tech"}}
        ]
    }
)

# Text search with filter
results = collection.search(
    text="programming",
    index="content_search",
    k=10,
    where={"author": "alice"}
)
```

### 3.4 Hybrid Search

```python
# Combine vector and text search
results = collection.search(
    vector=query_embedding,
    vector_index="vec_hnsw",
    text="machine learning",
    text_index="content_search",
    k=10,
    fusion="rrf",                    # "rrf" | "weighted"
    fusion_params={"k": 60},         # RRF constant
    where={"year": {"$gte": 2020}}
)

# Weighted fusion with explicit weights
results = collection.search(
    vector=query_embedding,
    vector_index="vec_hnsw",
    text="machine learning",
    text_index="content_search", 
    k=10,
    fusion="weighted",
    fusion_params={
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "normalize": True            # normalize scores before fusion
    }
)
```

### 3.5 Filter DSL Reference

| Operator | Example | Description |
|----------|---------|-------------|
| `$eq` | `{"field": "value"}` | Equals (implicit) |
| `$ne` | `{"field": {"$ne": "x"}}` | Not equals |
| `$gt` | `{"year": {"$gt": 2020}}` | Greater than |
| `$gte` | `{"year": {"$gte": 2020}}` | Greater than or equal |
| `$lt` | `{"price": {"$lt": 100}}` | Less than |
| `$lte` | `{"price": {"$lte": 100}}` | Less than or equal |
| `$in` | `{"tag": {"$in": ["a","b"]}}` | Value in list |
| `$nin` | `{"tag": {"$nin": ["x"]}}` | Value not in list |
| `$contains` | `{"tags": {"$contains": "tech"}}` | Array contains value |
| `$and` | `{"$and": [{...}, {...}]}` | All conditions match |
| `$or` | `{"$or": [{...}, {...}]}` | Any condition matches |

### 3.6 Get and Delete

```python
# Get by ID
docs = collection.get(ids=[1, 2])

# Get with filter
docs = collection.get(
    where={"author": "alice"},
    limit=100,
    offset=0
)

# Delete by ID
collection.delete(ids=[1])

# Delete with filter
collection.delete(where={"year": {"$lt": 2020}})

# Update metadata
collection.update(
    ids=[1],
    metadatas=[{"year": 2025}]  # partial update
)
```

---

## 4. Storage Layout

### 4.1 Document Store

Documents are stored in pages optimized for sequential scan and point lookup. Content and metadata are stored together in the same page space.

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Storage Layout                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Metadata Page (Page 0)              │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ magic: u64           │ version: u32  │ flags: u32    │   │
│  │ doc_count: u64       │ schema_page: PID              │   │
│  │ id_index_page: PID   │ free_list_page: PID           │   │
│  │ next_doc_id: u64     │ vector_dim: u32               │   │
│  │ distance_metric: u8  │ reserved: u8[7]               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Schema Page (Page 1)                │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ field_count: u16                                      │   │
│  │ fields[]: { name: char[64], type: u8, flags: u8 }    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DOC ID Index Pages (B-tree on DOC ID)            │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Maps integer doc_id → doc_page_ptr                   │   │
│  │   where doc_page_ptr = (page_id: PID, slot: u16)     │   │
│  │                                                       │   │
│  │ Enables O(log n) point lookup by doc_id              │   │
│  │ Note: Vectors are NOT stored here - they live in     │   │
│  │       the vector index itself                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Document Pages                      │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Page Header:                                          │   │
│  │   dirty: bool │ doc_count: u16 │ free_space: u16     │   │
│  │   next_page: PID │ prev_page: PID                    │   │
│  │                                                       │   │
│  │ Slot Directory (grows down from end):                │   │
│  │   slots[]: { offset: u16, length: u16, flags: u8 }   │   │
│  │     flags: 0x01 = has_overflow (continuation chain)  │   │
│  │                                                       │   │
│  │ Document Records (grows up from header):             │   │
│  │   ┌─────────────────────────────────────────────┐    │   │
│  │   │ doc_id: u64                                 │    │   │
│  │   │ total_length: u32 (full doc size)           │    │   │
│  │   │ content_length: u32                         │    │   │
│  │   │ content: char[] (inline or first chunk)     │    │   │
│  │   │ metadata_length: u32                        │    │   │
│  │   │ metadata: msgpack[] (inline or continues)   │    │   │
│  │   │ overflow_page: PID (if total > page space)  │    │   │
│  │   └─────────────────────────────────────────────┘    │   │
│  │                                                       │   │
│  │ Variable-length support:                              │   │
│  │   • Small docs (<12KB): fit entirely in one page     │   │
│  │   • Large docs (>=12KB): chain to overflow pages     │   │
│  │   • Overflow chain: linked list within same file     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 Overflow Pages                        │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ For documents exceeding page size:                   │   │
│  │                                                       │   │
│  │ Overflow Page Header:                                 │   │
│  │   dirty: bool │ parent_doc_id: u64                   │   │
│  │   continuation_length: u32                           │   │
│  │   next_overflow: PID (0 = end of chain)             │   │
│  │                                                       │   │
│  │ Continuation Data:                                    │   │
│  │   data: bytes[] (remaining content/metadata)         │   │
│  │                                                       │   │
│  │ Reading large doc: follow overflow chain             │   │
│  │   doc_page → overflow_page_1 → ... → overflow_page_N │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Design Rationale**:
- **Integer doc_id**: Efficient B-tree indexing and dense array mapping in vector indices
- **Slotted pages**: Enable variable-size documents with in-place updates
- **Content + metadata together**: Co-located for single read; both stored in document pages
- **Inline small docs**: Documents < 12KB fit in one page for fast access
- **Overflow chain**: Large documents chain to overflow pages in same file space
- **No separate vector storage**: Vectors stored directly in vector index files (HNSW/DiskANN/IVF-PQ)
- **ID index → document only**: B-tree maps doc_id to document location; vector access via vector index

### 4.2 Vector Index Storage

Each vector index is stored in its own file with index-specific layout. **Vectors are stored directly in the index** and accessed by integer doc_id.

```
┌─────────────────────────────────────────────────────────────┐
│                    HNSW Index File Layout                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Metadata Page (Page 0)              │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ magic: u64           │ version: u32                  │   │
│  │ collection_id: u32   │ index_id: u32                 │   │
│  │ dim: u32             │ M: u32 │ ef_construction: u32 │   │
│  │ node_count: u64      │ entry_point: u32              │   │
│  │ max_level: u32       │ base_page: PID                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Graph Node Pages                    │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Fixed-size nodes: NodesPerPage = PageSize / NodeSize │   │
│  │                                                       │   │
│  │ Node Layout (doc_id maps to node position):          │   │
│  │   ┌───────────────────────────────────────────────┐  │   │
│  │   │ doc_id: u64 (document identifier)             │  │   │
│  │   │ vector: float[dim] (STORED HERE)              │  │   │
│  │   │ level: u8                                     │  │   │
│  │   │ neighbors_per_level[]: { count: u16,          │  │   │
│  │   │                         ids: u64[M or M0] }   │  │   │
│  │   └───────────────────────────────────────────────┘  │   │
│  │                                                       │   │
│  │ Vector Access Pattern (existing HNSW/DiskANN style): │   │
│  │   • doc_id → node_index (via internal mapping)       │   │
│  │   • page_id = base_page + (node_index / NodesPerPage)│   │
│  │   • slot = node_index % NodesPerPage                 │   │
│  │   • vector = page.nodes[slot].vector                 │   │
│  │                                                       │   │
│  │ No separate vector storage needed - vectors live     │   │
│  │ with graph structure for cache locality              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- Vectors stored **directly in index file**, not in document storage
- Access pattern: `doc_id` → vector index node → vector data
- Same layout as existing HNSW/DiskANN/IVF-PQ implementations
- Document storage contains content+metadata only; vector indices contain vectors+graph

### 4.3 Text Index Storage

```
┌─────────────────────────────────────────────────────────────┐
│                    Text Index File Layout                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Metadata Page                       │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ magic: u64           │ version: u32                  │   │
│  │ collection_id: u32   │ index_id: u32                 │   │
│  │ vocab_size: u64      │ doc_count: u64                │   │
│  │ avg_doc_len: f32     │ dict_root_page: PID           │   │
│  │ analyzer_type: u8    │ language: u8                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Term Dictionary (B-tree on term)           │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Maps term → { doc_freq, posting_list_page }          │   │
│  │ Sorted for prefix/range queries                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Posting List Pages                  │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Per term, sorted by internal_id:                     │   │
│  │                                                       │   │
│  │ Posting Entry:                                        │   │
│  │   ┌───────────────────────────────────────────────┐  │   │
│  │   │ internal_id: u32 (delta-encoded)              │  │   │
│  │   │ term_freq: u16                                │  │   │
│  │   │ positions[]: u16[] (optional, for phrase)     │  │   │
│  │   └───────────────────────────────────────────────┘  │   │
│  │                                                       │   │
│  │ Compression: VarInt delta encoding for IDs           │   │
│  │ Skip pointers every 128 entries for fast seek        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Document Norms Page                      │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ doc_lengths[internal_id] → u16 (for BM25)            │   │
│  │ Dense array for O(1) lookup                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Metadata Index Storage

Caliby allows users to build indices on metadata attributes for efficient filtering. **Important constraints**:
- **Metadata cannot be nested** (only one layer of key-value pairs)
- Each metadata index maps attribute values to document IDs
- Supports **single-field** and **composite (multi-field)** indices
- Composite indices support **leftmost prefix queries** (like MySQL secondary indices)

#### Composite Index and Leftmost Prefix Rule

A composite index on multiple fields `(field1, field2, field3)` can efficiently answer queries on:
- `field1` alone ✓
- `field1` AND `field2` ✓
- `field1` AND `field2` AND `field3` ✓
- `field2` alone ✗ (cannot skip leftmost field)
- `field1` AND `field3` ✗ (cannot skip middle field for range/equality)

This follows the **leftmost prefix rule** from MySQL/PostgreSQL secondary indices.

```
┌─────────────────────────────────────────────────────────────┐
│                  Metadata Index File Layout                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Metadata Page                       │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ magic: u64           │ version: u32                  │   │
│  │ collection_id: u32   │ index_id: u32                 │   │
│  │ field_count: u8      │ unique: bool                  │   │
│  │ fields[]: { name: char[64], type: u8 }               │   │
│  │ root_page: PID       │ height: u32                   │   │
│  │ entry_count: u64                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  B-tree Internal Pages                │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ key_count: u16                                        │   │
│  │ keys[]: { composite_key: (val1, val2, ...) }         │   │
│  │ children[]: PID (key_count + 1 children)             │   │
│  │                                                       │   │
│  │ Composite Key Ordering:                               │   │
│  │   Compare field1 first, then field2, then field3...  │   │
│  │   (lexicographic ordering on tuple)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   B-tree Leaf Pages                   │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ entry_count: u16 │ next_leaf: PID │ prev_leaf: PID   │   │
│  │                                                       │   │
│  │ entries[]:                                            │   │
│  │   ┌───────────────────────────────────────────────┐  │   │
│  │   │ composite_key: (val1, val2, ...) │ doc_id: u64│  │   │
│  │   └───────────────────────────────────────────────┘  │   │
│  │                                                       │   │
│  │ For non-unique keys: key includes doc_id suffix      │   │
│  │ For unique keys: direct key → doc_id mapping         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Metadata Index Operations

```python
# Create single-field metadata index
articles.create_metadata_index(
    name="author_idx",
    fields=["author"],      # Single field
    unique=False            # Multiple docs can have same author
)

# Create composite metadata index (like MySQL secondary index)
articles.create_metadata_index(
    name="category_year_idx",
    fields=["category", "year"],  # Composite: (category, year)
    unique=False
)

# This index can efficiently answer:
# - WHERE category = 'tech'                    (uses index) ✓
# - WHERE category = 'tech' AND year = 2024   (uses index) ✓
# - WHERE year = 2024                          (full scan)  ✗

# Create unique composite index
articles.create_metadata_index(
    name="isbn_idx",
    fields=["isbn"],
    unique=True  # Each ISBN is unique
)

# Create 3-field composite index for complex filtering
products.create_metadata_index(
    name="store_category_price_idx",
    fields=["store_id", "category", "price"],
    unique=False
)
```

#### Leftmost Prefix Query Examples

```python
# Index: (category, year, author)

# ✓ Uses index - matches leftmost prefix (category)
results = collection.get(where={"category": "tech"})

# ✓ Uses index - matches leftmost prefix (category, year)
results = collection.get(where={
    "$and": [
        {"category": "tech"},
        {"year": {"$gte": 2020}}
    ]
})

# ✓ Uses index - matches full prefix (category, year, author)
results = collection.search_vector(
    query_vec,
    index="vec_hnsw",
    k=10,
    where={
        "$and": [
            {"category": "tech"},
            {"year": 2024},
            {"author": "alice"}
        ]
    }
)

# ✗ Cannot use index efficiently - skips leftmost field
# Falls back to full scan or uses different index
results = collection.get(where={"year": 2024})

# ✗ Cannot use index for range on middle field after equality
# Uses index for (category), post-filters (author)
results = collection.get(where={
    "$and": [
        {"category": "tech"},
        {"author": "alice"}  # Skipped year, can't use index for this
    ]
})
```

#### Metadata Update with Index Sync

When updating metadata, B-tree indices are automatically updated:

```python
# Update metadata - B-tree indices are automatically synchronized
collection.update(
    ids=[1, 2, 3],
    metadatas=[
        {"author": "new_author", "year": 2025},  # Partial update
        {"author": "alice"},
        {"tags": ["new_tag"]}
    ]
)
```

### 4.5 File Organization

Following the current Caliby catalog system conventions:

```
data_directory/
├── caliby_catalog                            # Catalog file (index registry)
├── .caliby.lock                              # Directory lock for single-process access
├── caliby_collection_1_articles.dat          # Collection 1 document storage
├── caliby_collection_2_embeddings.dat        # Collection 2 document storage
├── caliby_hnsw_10_vec_hnsw.dat              # HNSW index (id=10, contains vectors)
├── caliby_diskann_11_vec_diskann.dat         # DiskANN index (id=11, contains vectors)
├── caliby_text_12_content_search.dat         # Text index (id=12)
└── caliby_btree_13_year_idx.dat              # B-tree index (id=13)
```

**Naming Convention** (from `MultiFileStorage::make_index_filename`):
```
caliby_{type}_{id}_{name}.dat
```
Where:
- `type`: `hnsw`, `diskann`, `ivf`, `text`, `btree`, `collection`
- `id`: Numeric index/collection ID allocated by catalog
- `name`: User-provided name
- Extension: `.dat` (all files use same extension)

**Key Points**:
- **Flat directory structure**: All files in same data directory, no subdirectories
- **Catalog-driven naming**: Index ID and type prefix enable programmatic file discovery
- **Lock file**: `.caliby.lock` ensures single-process access to prevent corruption
- **Collection files**: Store document pages (content + metadata + overflow chains)
- **Index files**: Self-contained (vector indices include vectors + graph structure)
- **O_DIRECT**: Index files opened with `O_DIRECT` for buffer pool control (except catalog)

---

## 5. Filtered Search Strategy

### 5.1 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Filtered Search Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Parse Filter → Build Filter Plan                        │
│     ┌─────────────────────────────────────────────────┐     │
│     │ where={"$and": [{"year": {"$gte": 2020}},       │     │
│     │                 {"tags": {"$contains": "tech"}}]}│     │
│     └─────────────────────────────────────────────────┘     │
│                         │                                    │
│                         ▼                                    │
│  2. Index Selection + Selectivity Estimation                │
│     ┌─────────────────────────────────────────────────┐     │
│     │ • year_idx (B-tree): estimate ~30% selectivity  │     │
│     │ • tags: no index, must scan                     │     │
│     │ • Combined: ~15% estimated                      │     │
│     └─────────────────────────────────────────────────┘     │
│                         │                                    │
│          ┌──────────────┴──────────────────┐                │
│          ▼                                 ▼                │
│   Selectivity < 5%                  Selectivity >= 5%       │
│          │                                 │                │
│          ▼                                 ▼                │
│  ┌───────────────────┐          ┌───────────────────┐      │
│  │  PRE-FILTER MODE  │          │ INLINE-FILTER MODE│      │
│  ├───────────────────┤          ├───────────────────┤      │
│  │ 1. Eval filter    │          │ 1. Start ANN      │      │
│  │    → bitmap       │          │ 2. For each cand: │      │
│  │ 2. If small:      │          │    - Check filter │      │
│  │    exact KNN      │          │    - Skip if fail │      │
│  │ 3. If medium:     │          │ 3. Over-fetch 2x  │      │
│  │    ANN on subset  │          │ 4. Early stop at k│      │
│  └───────────────────┘          └───────────────────┘      │
│          │                                 │                │
│          └──────────────┬──────────────────┘                │
│                         ▼                                    │
│  3. Return top-k results with scores                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Index-Accelerated Filtering

When metadata indices exist on filter fields:

```
Filter: {"category": "tech", "year": {"$gte": 2020}, "author": "alice"}

With composite index (category, year):
  • category_year_idx.range_scan(("tech", 2020), ("tech", ∞)) → bitmap1
  • Apply author == "alice" as post-filter

With separate indices:
  • category_idx.lookup("tech") → bitmap1
  • year_idx.range_scan(2020, ∞) → bitmap2
  • author_idx.lookup("alice") → bitmap3 (if exists)
  • result = bitmap1 AND bitmap2 AND bitmap3

Leftmost Prefix Matching:
  Index (category, year, author) can serve:
  • WHERE category = 'x'                         → prefix scan on (category)
  • WHERE category = 'x' AND year = 2024        → prefix scan on (category, year)
  • WHERE category = 'x' AND year >= 2020       → range scan on (category, year)
  • WHERE year = 2024                            → CANNOT use index (skips leftmost)
```

### 5.3 Query Optimizer

Caliby includes a simple cost-based query optimizer that selects the optimal index for filter evaluation, with special handling for composite indices and leftmost prefix matching.

#### Optimizer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Optimizer                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: FilterCondition + Available Indices                 │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              1. Analyze Filter Structure              │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ • Decompose $and/$or into leaf predicates            │   │
│  │ • Identify indexed vs non-indexed fields             │   │
│  │ • Extract equality, range predicates                 │   │
│  │ • Match predicates to composite index prefixes       │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              2. Composite Index Matching              │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ For each composite index (f1, f2, f3, ...):          │   │
│  │                                                       │   │
│  │ • Find longest usable prefix:                        │   │
│  │   - f1 must have equality or range predicate         │   │
│  │   - f2 usable only if f1 has equality predicate      │   │
│  │   - f3 usable only if f1,f2 have equality predicates │   │
│  │   - First range predicate ends the usable prefix     │   │
│  │                                                       │   │
│  │ Example: Index (category, year, author)              │   │
│  │   Filter: category='tech' AND year>=2020 AND author='alice'  │
│  │   → Usable prefix: (category, year) - 2 fields       │   │
│  │   → author='alice' becomes post-filter               │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              3. Cost Estimation                       │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ For each candidate index:                            │   │
│  │   • Estimate selectivity based on prefix length      │   │
│  │   • Longer prefix = more selective = lower cost      │   │
│  │   • Calculate index scan cost                        │   │
│  │                                                       │   │
│  │ Selectivity heuristics:                              │   │
│  │   • 1-field equality:  ~10%                          │   │
│  │   • 2-field equality:  ~1%                           │   │
│  │   • 3-field equality:  ~0.1%                         │   │
│  │   • Range on last field: multiply by ~30%            │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              4. Plan Selection                        │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ Choose plan with lowest estimated cost:              │   │
│  │                                                       │   │
│  │ Option A: Single Composite Index Scan                │   │
│  │   → Use index with longest matching prefix           │   │
│  │   → Apply remaining predicates as post-filter        │   │
│  │                                                       │   │
│  │ Option B: Multi-Index Intersection                   │   │
│  │   → Use multiple single-field indices                │   │
│  │   → Intersect results, then apply post-filters       │   │
│  │                                                       │   │
│  │ Option C: Full Scan                                  │   │
│  │   → When no suitable index or low selectivity gain   │   │
│  └──────────────────────────────────────────────────────┘   │
│                         │                                    │
│                         ▼                                    │
│  Output: QueryPlan with chosen index and scan bounds        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Query Plan Structure

```cpp
struct QueryPlan {
    // Primary index to use (nullptr = full scan)
    MetadataIndex* primary_index;
    
    // Index operation type
    enum class IndexOp {
        FULL_SCAN,           // No index, scan all documents
        POINT_LOOKUP,        // Equality on all index fields
        PREFIX_SCAN,         // Equality on prefix, scan suffix
        RANGE_SCAN,          // Range on last usable field
        MULTI_INDEX          // Intersection of multiple indices
    } op;
    
    // Number of index fields used (for composite indices)
    size_t prefix_length;
    
    // Scan bounds (for composite key)
    std::vector<Value> lower_bound;  // e.g., ("tech", 2020)
    std::vector<Value> upper_bound;  // e.g., ("tech", INT_MAX)
    bool lower_inclusive;
    bool upper_inclusive;
    
    // For multi-index plans
    std::vector<std::pair<MetadataIndex*, FilterCondition>> index_scans;
    
    // Remaining predicates to apply after index scan
    std::vector<FilterCondition> post_filters;
    
    // Estimated cost and selectivity
    float estimated_cost;
    float estimated_selectivity;
};
```

#### Optimizer API

```python
# The optimizer is invoked automatically during filtered search
results = collection.search_vector(
    query_vec,
    index="vec_hnsw",
    k=10,
    where={
        "$and": [
            {"category": "tech"},       # Uses composite index prefix
            {"year": {"$gte": 2020}},   # Range on second field
            {"rating": {"$gt": 4.0}}    # Post-filter (not in index)
        ]
    }
)

# With index (category, year, author):
# Optimizer selects:
# 1. Usable prefix: (category, year) - 2 fields
# 2. Range scan: ("tech", 2020) to ("tech", MAX_INT)
# 3. Post-filter: rating > 4.0
```

#### Composite Index Selection Rules

| Query Pattern | Index (f1, f2, f3) Usage | Notes |
|---------------|-------------------------|-------|
| `f1 = x` | Uses 1-field prefix | Full index on f1 |
| `f1 = x AND f2 = y` | Uses 2-field prefix | Best selectivity |
| `f1 = x AND f2 >= y` | Uses 2-field prefix (range) | Range on f2 |
| `f1 = x AND f3 = z` | Uses 1-field prefix only | f2 skipped, f3 post-filter |
| `f2 = y` | Cannot use index | Leftmost field missing |
| `f1 >= x` | Uses 1-field prefix (range) | Range stops further prefix use |
| `f1 = x AND f2 = y AND f3 = z` | Uses full 3-field prefix | Most selective |

#### Index Selection Rules

| Predicate Type | Index Requirement | Selection Priority |
|----------------|-------------------|-------------------|
| Composite equality | Metadata index on fields | Highest (most selective) |
| Single-field `$eq` | Metadata index on field | High |
| `$in` | Metadata index on field | High (union of lookups) |
| `$gt/$gte/$lt/$lte` | Metadata index on field | Medium |
| `$contains` (array) | No index support | Low (post-filter) |
| `$ne`, `$nin` | Poor index fit | Lowest (post-filter) |

### 5.4 Hybrid Search Fusion

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Search Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: vector, text, filter, k                             │
│                                                              │
│  ┌─────────────────┐              ┌─────────────────┐       │
│  │  Vector Search  │              │   Text Search   │       │
│  │  (with filter)  │              │  (with filter)  │       │
│  │  top-2k results │              │  top-2k results │       │
│  └────────┬────────┘              └────────┬────────┘       │
│           │                                │                 │
│           ▼                                ▼                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Score Fusion                       │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                      │    │
│  │  RRF (Reciprocal Rank Fusion):                      │    │
│  │    score(d) = Σ 1/(k + rank_i(d))                   │    │
│  │    where k=60 (constant), rank_i = rank in list i   │    │
│  │                                                      │    │
│  │  Weighted (requires normalization):                  │    │
│  │    score(d) = α·norm(vec_score) + (1-α)·norm(bm25)  │    │
│  │    norm(s) = (s - min) / (max - min)                │    │
│  │                                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│           │                                                  │
│           ▼                                                  │
│  Sort by fused score → Return top-k                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Catalog Schema

### 6.1 Catalog Tables (Conceptual)

```sql
-- Collections table
CREATE TABLE collections (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(256) UNIQUE NOT NULL,
    schema_json     TEXT NOT NULL,           -- JSON schema definition
    vector_dim      INT,                     -- NULL if no vector support
    distance_metric VARCHAR(16),             -- 'cosine', 'l2', 'ip'
    doc_count       BIGINT DEFAULT 0,
    created_at      TIMESTAMP,
    updated_at      TIMESTAMP
);

-- Indices table  
CREATE TABLE indices (
    id              SERIAL PRIMARY KEY,
    collection_id   INT REFERENCES collections(id),
    name            VARCHAR(256) NOT NULL,
    type            VARCHAR(32) NOT NULL,    -- 'hnsw', 'diskann', 'text', 'btree'
    config_json     TEXT NOT NULL,           -- Index-specific config
    file_path       VARCHAR(512) NOT NULL,
    status          VARCHAR(16) NOT NULL,    -- 'building', 'ready', 'error'
    entry_count     BIGINT DEFAULT 0,
    created_at      TIMESTAMP,
    updated_at      TIMESTAMP,
    UNIQUE(collection_id, name)
);
```

### 6.2 Catalog File Format

```
┌─────────────────────────────────────────────────────────────┐
│                    catalog.meta Layout                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Header:                                                     │
│    magic: "CALIBY01"                                        │
│    version: u32                                              │
│    collection_count: u32                                     │
│    index_count: u32                                          │
│                                                              │
│  Collection Entries[]:                                       │
│    id: u32                                                   │
│    name_len: u16 | name: char[]                             │
│    schema_len: u32 | schema_json: char[]                    │
│    vector_dim: u32 (0 if none)                              │
│    distance_metric: u8                                       │
│    doc_count: u64                                            │
│    data_dir: char[256]                                       │
│                                                              │
│  Index Entries[]:                                            │
│    id: u32                                                   │
│    collection_id: u32                                        │
│    name_len: u16 | name: char[]                             │
│    type: u8                                                  │
│    config_len: u32 | config_json: char[]                    │
│    file_path: char[256]                                      │
│    status: u8                                                │
│    entry_count: u64                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Common Query Patterns & Optimizations

### 7.1 RAG Retrieval (Most Common)

```python
# Typical RAG query: semantic search + recency filter
results = collection.search(
    vector=embed(user_query),
    index="vec_hnsw",
    k=5,
    where={"created_at": {"$gte": last_week}}
)
```

**Optimization**: B-tree on `created_at` enables fast bitmap generation.

### 7.2 Hybrid RAG with Keyword Boost

```python
# User query contains specific terms that should boost relevance
results = collection.search(
    vector=embed(user_query),
    vector_index="vec_hnsw",
    text=extract_keywords(user_query),  # "python async await"
    text_index="content_search",
    k=5,
    fusion="rrf"
)
```

**Optimization**: RRF naturally handles different score scales.

### 7.3 Filtered Faceted Search

```python
# E-commerce: find similar products in category with price range
results = collection.search(
    vector=product_embedding,
    index="vec_hnsw",
    k=20,
    where={
        "$and": [
            {"category": "electronics"},
            {"price": {"$gte": 100, "$lte": 500}},
            {"in_stock": True}
        ]
    }
)
```

**Optimization**: 
- Metadata index on `(category, price)` for composite queries
- Index on `category` for equality → fast prefix lookup
- Range extension on `price` within same index scan
- Bitmap intersection before ANN

### 7.4 Full-Text Search with Metadata

```python
# Search articles by content with author filter (no vector)
results = collection.search(
    text="machine learning optimization",
    index="content_search",
    k=10,
    where={"author": {"$in": ["alice", "bob", "charlie"]}}
)
```

**Optimization**: Text-only collection, no vector overhead.

---

## 8. Future Considerations

1. **Approximate Metadata Filters**: Bloom filters for high-cardinality fields
2. **Streaming/Incremental Index Updates**: Avoid full rebuild on insert
3. **Index Advisor**: Suggest indices based on query patterns
4. **Predicate-Aware ANN**: ACORN-style graph augmentation for stable filtered recall
5. **Partial Index Support**: Indices on subset of documents matching condition
6. **Index-Only Scans**: Return results directly from index without document lookup
