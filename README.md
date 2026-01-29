# Caliby 🚀

**High-Performance Embeddable Vector Database with Document Storage, Hybrid Search, and Filtering**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Linux](https://img.shields.io/badge/platform-linux-lightgrey.svg)](https://www.linux.org/)

Caliby is a high-performance embeddable vector database that combines document storage, semantic search, full-text search, and metadata filtering in a single library. Built on an innovative buffer pool architecture, Caliby efficiently handles datasets larger than available memory while delivering **in-memory speed when data fits in RAM** and **graceful degradation when it doesn't** — no expensive hardware or distributed systems required.

## ✨ Key Features

- **📚 Document Storage**: Store vectors, text, and metadata with flexible schemas
- **🔍 Filtered Search**: Efficient vector search with metadata filtering
- **🔗 Hybrid Search**: Combine vector similarity and BM25 full-text search
- **🔥 In-Memory Speed**: Matches or exceeds HNSWLib/Faiss/Usearch when data fits in RAM
- **💾 Larger-Than-Memory**: Seamless performance with datasets exceeding available memory
- **🎯 Multiple Index Types**: Inverted Index, B+tree, HNSW, DiskANN, and IVF+PQ algorithms
- **🔧 Embeddable**: Single-process library, no server required

## 🚀 Quick Start

### Prerequisites

Caliby requires the following system dependencies:
- C++17 compatible compiler (GCC 9+ or Clang 10+)
- CMake 3.15+
- OpenMP
- Abseil C++ library
- Python 3.8+

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libomp-dev libabsl-dev python3-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install -y gcc-c++ cmake libomp-devel abseil-cpp-devel python3-devel
```

### Installation

**From PyPI (Recommended):**
```bash
pip install caliby
```

**From Source:**
```bash
git clone --recursive https://github.com/zxjcarrot/caliby.git
cd caliby
pip install -e .
```

**Note:** The `--recursive` flag is required to initialize the pybind11 submodule. If you already cloned without it, run:
```bash
git submodule update --init --recursive
```

### Collection API (Recommended)

The Collection API provides a high-level interface for storing documents with vectors, text, and metadata:

```python
import caliby
import numpy as np

# Initialize and create a collection
caliby.set_buffer_config(size_gb=1.0)
caliby.open('/tmp/my_database')
collection = caliby.create_collection("products")

# Define schema
collection.set_schema({
    "embedding": {"type": "vector", "dim": 128},
    "description": {"type": "text"},
    "category": {"type": "metadata"}
})

# Add documents
collection.add_documents([
    {"id": "1", "embedding": np.random.rand(128).astype('float32'),
     "description": "Wireless headphones", "category": "electronics"},
    {"id": "2", "embedding": np.random.rand(128).astype('float32'),
     "description": "Running shoes", "category": "sports"}
])

# Create indices
collection.create_hnsw_index("embedding", m=16, ef_construction=200)
collection.create_text_index("description")
collection.create_metadata_index("category")

# Vector search with filter (99.5% recall)
query = np.random.rand(128).astype('float32')
results = collection.search_vector("embedding", query, k=10, 
                                   filter={"category": "electronics"})

# Hybrid search (vector + text)
results = collection.search_hybrid("embedding", query, 
                                   text_field="description",
                                   text_query="wireless", k=10, alpha=0.5)

caliby.close()
```

📖 **See [docs/COLLECTION_API.md](docs/COLLECTION_API.md) for complete documentation** including advanced filtering, best practices, and performance tuning.

### Low-Level Index API

For direct control over indices:

```python
import caliby
import numpy as np

# Initialize the system and configure buffer pool
caliby.set_buffer_config(size_gb=1.0)  # Set buffer pool size
caliby.open('/tmp/caliby_data')  # Initialize catalog

# Create an HNSW index
index = caliby.HnswIndex(
    max_elements=1_000_000,     # Maximum number of vectors
    dim=128,                    # Vector dimension
    M=16,                       # HNSW parameter (connections per node)
    ef_construction=200,        # Construction-time search depth
    enable_prefetch=True,       # Enable prefetching for performance
    skip_recovery=False,        # Whether to skip recovery from disk
    index_id=0,                 # Unique index identifier for multi-index
    name='user_embeddings',     # Optional human-readable name
)

# Add vectors (batch)
vectors = np.random.rand(10000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)  # Parallel insertion

# Get index info
print(f"Index name: {index.get_name()}")  # Output: 'user_embeddings'
print(f"Dimension: {index.get_dim()}")

# Search (single query)
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, ef_search_param=50)

# Batch search (parallel)
queries = np.random.rand(100, 128).astype(np.float32)
results = index.search_knn_parallel(queries, k=10, ef_search_param=50, num_threads=4)

# Close when done
caliby.close()
```
## 🏗️ Index Types

### HNSW (Hierarchical Navigable Small World)

Best for: High recall requirements, moderate to large dataset sizes

```python
import caliby
import numpy as np

# Initialize system
caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,                    # Higher = better recall, more memory
    ef_construction=200,     # Higher = better graph quality, slower build
    enable_prefetch=True,    # Enable prefetching
    skip_recovery=False,
    index_id=0,              # Unique ID for multi-index support
    name='my_vectors',       # Optional human-readable name
)

# Add points
vectors = np.random.rand(100000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

# Search with ef_search_param
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, ef_search_param=100)
```

### DiskANN (Vamana Graph)

Best for: Filtered search, dynamic updates, very large graphs with tags/labels

```python
import caliby
import numpy as np

# Initialize system
caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

# Create DiskANN index
index = caliby.DiskANN(
    dimensions=128,
    max_elements=1_000_000,
    R_max_degree=64,    # Max graph degree (R)
    is_dynamic=True     # Enable dynamic inserts/deletes
)

# Build index with tags for filtering
vectors = np.random.rand(100000, 128).astype(np.float32)
tags = [[i % 100] for i in range(100000)]  # Tags for filtering

params = caliby.BuildParams()
params.L_build = 100       # Build-time search depth
params.alpha = 1.2         # Alpha parameter for Vamana
params.num_threads = 4

index.build(vectors, tags, params)

# Search with params
search_params = caliby.SearchParams(L_search=50)
search_params.beam_width = 4

query = np.random.rand(128).astype(np.float32)
labels, distances = index.search(query, K=10, params=search_params)

# Filtered search (only return vectors with specific tag)
labels, distances = index.search_with_filter(query, filter_label=42, K=10, params=search_params)

# Dynamic operations (if is_dynamic=True)
new_point = np.random.rand(128).astype(np.float32)
index.insert_point(new_point, tags=[99], external_id=100000)
index.lazy_delete(external_id=100000)
index.consolidate_deletes(params)
```

### IVF+PQ (Inverted File with Product Quantization)

Best for: Very large datasets (10M+ vectors), memory-constrained environments

```python
import caliby
import numpy as np

# Initialize system with buffer pool
caliby.set_buffer_config(size_gb=0.5)  # Small buffer for large datasets
caliby.open('/tmp/caliby_data')

index = caliby.IVFPQIndex(
    max_elements=10_000_000,
    dim=128,
    num_clusters=256,           # Number of IVF clusters (K)
    num_subquantizers=8,        # Number of PQ subquantizers (M), dim must be divisible by this
    retrain_interval=10000,     # Retrain centroids every N insertions
    skip_recovery=False,
    index_id=0,
    name='large_dataset'
)

# Train the index first (required for IVF+PQ)
training_data = np.random.rand(50000, 128).astype(np.float32)
index.train(training_data)

# Add points (after training)
vectors = np.random.rand(1000000, 128).astype(np.float32)
index.add_points(vectors, num_threads=4)

# Search with nprobe parameter
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, nprobe=8)
```

## 🔧 Advanced Configuration

### Multi-Index Support

Create and manage multiple independent indexes with unique IDs and names:

```python
import caliby
import numpy as np

# Initialize system once
caliby.set_buffer_config(size_gb=2.0)
caliby.open('/tmp/caliby_data')

# Create multiple indexes with unique IDs and names
user_index = caliby.HnswIndex(
    max_elements=100_000, dim=128, M=16, ef_construction=200,
    enable_prefetch=True, skip_recovery=True, index_id=1, name='user_embeddings'
)

product_index = caliby.HnswIndex(
    max_elements=200_000, dim=256, M=16, ef_construction=200,
    enable_prefetch=True, skip_recovery=True, index_id=2, name='product_embeddings'
)

# Access index by name
print(f"Working with: {user_index.get_name()}")
print(f"Dimension: {user_index.get_dim()}")

# Each index operates independently
user_vectors = np.random.rand(10000, 128).astype(np.float32)
product_vectors = np.random.rand(15000, 256).astype(np.float32)
user_index.add_points(user_vectors, num_threads=4)
product_index.add_points(product_vectors, num_threads=4)
```
### Persistence & Recovery

```python
import caliby

# Indexes are automatically persisted via the buffer pool
caliby.set_buffer_config(size_gb=1.0)
caliby.open('/path/to/caliby_data')  # Data directory for persistent storage

# Create index (will be persisted automatically)
index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,
    ef_construction=200,
    enable_prefetch=True,
    skip_recovery=False,  # Set to False to enable recovery
    index_id=1,
    name='my_index'
)

# Manual flush to ensure all data is written
index.flush()

# Recovery happens automatically when reopening with same directory
caliby.close()

# Later: reopen and recover
caliby.open('/path/to/caliby_data')
recovered_index = caliby.HnswIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,
    ef_construction=200,
    enable_prefetch=True,
    skip_recovery=False,  # Will recover existing index
    index_id=1,  # Must match original
    name='my_index'
)

if recovered_index.was_recovered():
    print("Index successfully recovered from disk!")
```

### Concurrent Access

```python
# Thread-safe by default
from concurrent.futures import ThreadPoolExecutor

def search_worker(query):
    return index.search(query, k=10)

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(search_worker, queries))
```

## 📁 Project Structure

```
caliby/
├── include/caliby/          # C++ headers
│   ├── calico.hpp           # Core buffer pool system
│   ├── hnsw.hpp             # HNSW index
│   ├── ivfpq.hpp            # IVF+PQ index
│   ├── diskann.hpp          # DiskANN index (experimental)
│   ├── catalog.hpp          # Index catalog management
│   └── distance.hpp         # Distance functions
├── src/                     # C++ implementation
│   ├── bindings.cpp         # Python bindings
│   ├── hnsw.cpp
│   ├── ivfpq.cpp
│   └── calico.cpp
├── examples/                # Usage examples
├── benchmark/               # Performance benchmarks
├── tests/                   # Python tests
└── third_party/             # Dependencies
    └── pybind11/            # Python binding library (submodule)
```

## 🛠️ Building from Source

### Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- GCC 10+ or Clang 12+
- CMake 3.16+
- Python 3.8+ with development headers
- libaio-dev

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake python3-dev libaio-dev

# Enable huge pages (recommended for performance)
sudo sysctl -w vm.nr_hugepages=1024
```

### Build

```bash
git clone https://github.com/zxjcarrot/caliby.git
cd caliby
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package
cd ..
pip install -e .
```

### Run Tests

```bash
# C++ tests
cd build && ctest --output-on-failure

# Python tests
pytest python/tests/
```

## 📚 Documentation

- **[Collection API Guide](docs/COLLECTION_API.md)** - High-level API for documents with vectors, text, and metadata
- **[Usage Guide](docs/USAGE.md)** - General usage patterns and examples
- **[Benchmarks](benchmark/README.md)** - Performance comparisons and benchmarking tools

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/zxjcarrot/caliby/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zxjcarrot/caliby/discussions)
- **Email**: xinjing@mit.edu

---

**⭐ If you find Caliby useful, please consider giving it a star!**
