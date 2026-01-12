# Caliby 🚀

**High-Performance Vector Search with Efficient Larger-Than-Memory Support**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Linux](https://img.shields.io/badge/platform-linux-lightgrey.svg)](https://www.linux.org/)

Caliby is a high-performance vector similarity search library that efficiently handles datasets larger than available memory. Built on an innovative buffer pool design, Caliby delivers **best-in-class in-memory performance when data fits in RAM** and **graceful degradation when it doesn't** — without requiring expensive hardware or complex distributed systems.

## ✨ Key Features

- **🔥 In-Memory Speed**: Matches or exceeds Faiss performance when data fits in memory
- **💾 Larger-Than-Memory**: Seamlessly handles datasets that exceed RAM with minimal performance loss
- **🎯 Multiple Index Types**: HNSW, DiskANN, and IVF indexes with unified API
- **🐍 Python First**: Native Python bindings with NumPy integration
- **🔧 Embeddable**: Single-process library, no server required

## 🚀 Quick Start

### Installation
Build from source:

```bash
git clone https://github.com/zxjcarrot/caliby.git
cd caliby
pip install -e .
```

### Basic Usage

```python
import caliby
import numpy as np

# Create an HNSW index
index = caliby.HNSWIndex(
    max_elements=1_000_000,     # Maximum number of vectors
    dim=128,                    # Vector dimension
    M=16,                       # HNSW parameter (connections per node)
    ef_construction=200,        # Construction-time search depth
    skip_recovery=False,        # Whether to skip recovery from disk
    index_id=0,                 # Unique index identifier for multi-index
    name='user_embeddings',     # Optional human-readable name
)

# Add vectors
vectors = np.random.rand(10000, 128).astype(np.float32)
index.add_points(vectors)

# Get index name
print(f"Index name: {index.get_name()}")  # Output: 'user_embeddings'

# Search
query = np.random.rand(128).astype(np.float32)
labels, distances = index.search_knn(query, k=10, ef_search=50)

# Batch search
queries = np.random.rand(100, 128).astype(np.float32)
results = index.search_knn_parallel(queries, k=10, ef_search=50, num_threads=4)
```
## 🏗️ Index Types

### HNSW (Hierarchical Navigable Small World)

Best for: High recall requirements, moderate dataset sizes

```python
index = caliby.HNSWIndex(
    max_elements=1_000_000,
    dim=128,
    M=16,                    # Higher = better recall, more memory
    ef_construction=200,     # Higher = better graph quality, slower build
    skip_recovery=False,
    index_id=0,              # Unique ID for multi-index support
    name='my_vectors',       # Optional human-readable name
)

# Search with ef_search parameter
labels, distances = index.search_knn(query, k=10, ef_search=100)
```

### DiskANN

Best for: Very large datasets, SSD-optimized access patterns

```python
index = caliby.DiskANNIndex(
    dim=128,
    max_elements=100_000_000,
    R=64,                    # Graph degree
    L=100,                   # Search list size
    alpha=1.2,               # Pruning parameter
)
```

### IVF (Inverted File Index)

Best for: Extremely large datasets, when some recall loss is acceptable

```python
index = caliby.IVFIndex(
    dim=128,
    max_elements=1_000_000_000,
    n_lists=4096,            # Number of clusters
    n_probe=32,              # Clusters to search
    quantizer='flat',        # 'flat', 'pq', 'sq'
)
```

## 🔧 Advanced Configuration

### Multi-Index Support

Create and manage multiple independent indexes with unique IDs and names:

```python
# Create multiple indexes with unique IDs and names
user_index = caliby.HNSWIndex(
    max_elements=100_000, dim=128, M=16, ef_construction=200,
    skip_recovery=True, index_id=1, name='user_embeddings'
)

product_index = caliby.HNSWIndex(
    max_elements=200_000, dim=256, M=16, ef_construction=200,
    skip_recovery=True, index_id=2, name='product_embeddings'
)

# Access index by name
print(f"Working with: {user_index.get_name()}")
print(f"Dimension: {user_index.get_dim()}")

# Each index operates independently
user_index.add_points(user_vectors)
product_index.add_points(product_vectors)
```
### Persistence & Recovery

```python
# Indexes are automatically persisted
index = caliby.HNSWIndex(
    storage_path='./my_index',
    sync_on_add=False,           # Batch writes for performance
)

# Manual checkpoint
index.checkpoint()

# Recovery happens automatically on load
index = caliby.HNSWIndex.load('./my_index')
print(f"Recovered {index.count()} vectors")
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
│   ├── buffer_pool.hpp      # Core buffer pool
│   ├── hnsw.hpp             # HNSW index
│   ├── diskann.hpp          # DiskANN index
│   ├── ivf.hpp              # IVF index
│   └── distance.hpp         # Distance functions
├── src/                     # C++ implementation
├── python/                  # Python bindings
│   ├── caliby/              # Python package
│   └── tests/               # Python tests
├── examples/                # Usage examples
├── benchmarks/              # Performance benchmarks
├── docs/                    # Documentation
└── tests/                   # C++ tests
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

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Performance Tuning](docs/performance-tuning.md)
- [Architecture Overview](docs/architecture.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on research from [Calico: High-Performance Buffer Management](link-to-paper)
- Inspired by [Faiss](https://github.com/facebookresearch/faiss), [HNSWlib](https://github.com/nmslib/hnswlib), and [DiskANN](https://github.com/microsoft/DiskANN)

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/zxjcarrot/caliby/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zxjcarrot/caliby/discussions)
- **Email**: xinjing@mit.edu

---

**⭐ If you find Caliby useful, please consider giving it a star!**
