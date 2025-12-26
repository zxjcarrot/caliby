# Contributing to Caliby

Thank you for your interest in contributing to Caliby! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to xinjing@mit.edu.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are ready for contributions
- Feel free to ask questions on any issue before starting work

### Reporting Bugs

Before creating a bug report:
1. Check existing issues to avoid duplicates
2. Collect information about your environment (OS, Python version, etc.)
3. Create a minimal reproduction case

Include in your bug report:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Relevant logs or error messages

### Suggesting Features

Feature requests are welcome! Please:
1. Check existing issues and discussions first
2. Describe the use case and motivation
3. Explain the proposed solution
4. Consider implementation complexity

## Development Setup

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    libaio-dev \
    git

# Optional: Enable huge pages for better performance
sudo sysctl -w vm.nr_hugepages=512
```

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/zxjcarrot/caliby.git
cd caliby

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCALIBY_BUILD_TESTS=ON
make -j$(nproc)
```

### IDE Setup

We recommend VS Code with these extensions:
- C/C++ (Microsoft)
- CMake Tools
- Python
- clangd (for better C++ intellisense)

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-ivf-pq-quantizer`
- `fix/memory-leak-in-hnsw`
- `docs/update-api-reference`
- `perf/optimize-distance-computation`

### Commit Messages

Follow conventional commits format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `perf`: Performance improvement
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Code Changes Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated if needed
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] No new warnings introduced

## Pull Request Process

### Before Submitting

1. Ensure your branch is up to date with `main`
2. Run the full test suite
3. Update documentation if needed
4. Add entry to CHANGELOG.md if applicable

### Submitting

1. Create a pull request against `main`
2. Fill out the PR template completely
3. Link related issues
4. Request review from maintainers

### Review Process

- Maintainers will review within 1-2 weeks
- Address feedback promptly
- Keep the PR focused and reasonably sized
- Large changes should be discussed in an issue first

## Coding Standards

### C++ Style

We follow a modified Google C++ style:

```cpp
// Use #pragma once for header guards
#pragma once

namespace caliby {

// Class names: PascalCase
class BufferPool {
public:
    // Method names: camelCase
    void allocatePage(PageId id);
    
    // Constants: kPascalCase
    static constexpr size_t kPageSize = 4096;
    
private:
    // Member variables: snake_case with trailing underscore
    size_t page_count_;
};

// Free functions: snake_case
void compute_distance(const float* a, const float* b, size_t dim);

}  // namespace caliby
```

### Python Style

We follow PEP 8 with these tools:
- Black for formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

```python
# Type hints are required for public APIs
def search(
    self,
    query: np.ndarray,
    k: int = 10,
    ef_search: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for nearest neighbors.
    
    Args:
        query: Query vector of shape (dim,) or (n_queries, dim)
        k: Number of nearest neighbors to return
        ef_search: Search depth (higher = better recall, slower)
        
    Returns:
        Tuple of (labels, distances) arrays
        
    Raises:
        ValueError: If query dimension doesn't match index dimension
    """
    ...
```

### Formatting

```bash
# C++ formatting
clang-format -i src/*.cpp include/caliby/*.hpp

# Python formatting
black python/
isort python/
```

## Testing

### C++ Tests

We use Google Test:

```cpp
#include <gtest/gtest.h>
#include "caliby/hnsw.hpp"

TEST(HNSWTest, BasicInsertAndSearch) {
    caliby::HNSWIndex index(128, 1000);
    
    std::vector<float> vec(128, 1.0f);
    index.add(vec.data(), 0);
    
    auto [labels, distances] = index.search(vec.data(), 1);
    EXPECT_EQ(labels[0], 0);
    EXPECT_FLOAT_EQ(distances[0], 0.0f);
}
```

Run C++ tests:
```bash
cd build
ctest --output-on-failure
```

### Python Tests

We use pytest:

```python
import pytest
import numpy as np
import caliby

class TestHNSWIndex:
    def test_basic_search(self):
        index = caliby.HNSWIndex(dim=128, max_elements=1000)
        
        vectors = np.random.rand(100, 128).astype(np.float32)
        index.add(vectors)
        
        labels, distances = index.search(vectors[0], k=1)
        assert labels[0] == 0
        assert distances[0] < 1e-6
    
    def test_persistence(self, tmp_path):
        index = caliby.HNSWIndex(
            dim=128, 
            max_elements=1000,
            storage_path=str(tmp_path / "test_index")
        )
        # ... test save/load
```

Run Python tests:
```bash
pytest python/tests/ -v --cov=caliby
```

### Performance Tests

For performance-related changes, include benchmark results:

```bash
python benchmarks/run_benchmarks.py --baseline main --compare your-branch
```

## Documentation

### Docstrings

All public APIs must have docstrings:

```python
def add(
    self,
    vectors: np.ndarray,
    ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Add vectors to the index.
    
    Args:
        vectors: Vectors to add, shape (n, dim)
        ids: Optional vector IDs. If None, IDs are auto-assigned.
        
    Returns:
        Array of assigned IDs
        
    Raises:
        ValueError: If vectors have wrong dimension
        RuntimeError: If index is full
        
    Example:
        >>> index = caliby.HNSWIndex(dim=128, max_elements=1000)
        >>> vectors = np.random.rand(100, 128).astype(np.float32)
        >>> ids = index.add(vectors)
        >>> print(f"Added {len(ids)} vectors")
    """
```

### Updating Documentation

- API changes require updated docstrings
- New features need documentation in `docs/`
- Update README.md for significant changes
- Add examples for new functionality

## Questions?

- Open a [Discussion](https://github.com/zxjcarrot/caliby/discussions)
- Ask on the related issue
- Email: xinjing@mit.edu

Thank you for contributing! 🙏
