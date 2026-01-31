# Caliby Open Source Contribution TODOs

This document outlines features and improvements that are open for contribution. If you're interested in contributing, please open an issue to discuss your approach before starting work.

## Table of Contents

1. [High Priority Features](#high-priority-features)
2. [Medium Priority Features](#medium-priority-features)
3. [Source Code TODOs](#source-code-todos)
4. [Good First Issues](#good-first-issues)
5. [Performance Improvements](#performance-improvements)

---

## High Priority Features

### 1. Update/Delete Support for Vector Indexes

**Status**: Not Implemented  
**Complexity**: High  
**Files**: `src/hnsw.cpp`, `src/diskann.cpp`, `src/ivfpq.cpp`

Currently, Caliby's vector indexes (HNSW, DiskANN, IVF-PQ) are append-only. Implementing update and delete operations would significantly improve usability for production workloads.

#### Requirements:

**Delete Operations:**
- Mark vectors as deleted without immediately reclaiming space (tombstone approach)
- Maintain a deleted vector bitmap or free list
- Skip deleted vectors during search operations
- Implement compaction/vacuum to reclaim space from deleted vectors
- Handle edge cases: deleting entry point in HNSW, deleting centroids in IVF

**Update Operations:**
- Support in-place updates when vector dimensions match
- Handle updates that require re-insertion (different neighbors in graph-based indexes)
- Maintain consistency between the vector data and graph structure
- Consider lazy vs eager update strategies

#### Reference Implementation:
- Look at hnswlib's delete implementation: https://github.com/nmslib/hnswlib
- Faiss IVF delete: https://github.com/facebookresearch/faiss

---

### 2. Vector Quantization Support

**Status**: Partial (IVF-PQ exists)  
**Complexity**: High  
**Files**: `include/caliby/ivfpq.hpp`, `src/ivfpq.cpp`, new files needed

Improve memory efficiency and search speed through additional quantization methods.

#### Product Quantization (PQ) Enhancements:
- [ ] Optimized PQ (OPQ) - rotation-optimized PQ for better accuracy
- [ ] Additive Quantization (AQ) - improved residual quantization
- [ ] Polysemous codes for faster distance computation

#### Scalar Quantization:
- [ ] INT8 quantization with calibration
- [ ] INT4/Binary quantization for extreme compression
- [ ] Automatic scale/zero-point computation

#### Implementation in IVF-PQ:
```cpp
// Current TODO in ivfpq.cpp:1639
// TODO: Trigger online retraining

// This should implement:
// 1. Detect distribution shift
// 2. Background retraining of centroids/codebooks
// 3. Gradual migration to new quantization
```

#### New Quantization Index Type:
```cpp
// Proposed: include/caliby/sq.hpp (Scalar Quantization)
template <typename T = int8_t>  // int8_t, int4_t, binary
class ScalarQuantizedIndex : public IndexBase {
    void train(const float* vectors, size_t n);
    void add(const float* vectors, size_t n, const uint64_t* labels);
    void search(const float* query, size_t k, uint64_t* labels, float* distances);
};
```

---

### 3. Inner Product Distance Implementation

**Status**: Placeholder exists  
**Complexity**: Medium  
**Files**: `include/caliby/ivfpq.hpp`, `include/caliby/distance.hpp`

```cpp
// Current TODO in ivfpq.hpp:22
using InnerProductDistance = hnsw_distance::SIMDAcceleratedL2;  // TODO: implement actual inner product
```

#### Requirements:
- Implement SIMD-accelerated inner product (MIPS - Maximum Inner Product Search)
- Handle normalization options (cosine similarity = normalized inner product)
- Support both AVX2 and AVX-512 code paths
- Integrate with IVF-PQ and HNSW indexes

---

## Medium Priority Features

### 4. B-Tree Inner Node Merging

**Status**: Not Implemented  
**Complexity**: Medium  
**Files**: `src/calico.cpp`

```cpp
// TODO at src/calico.cpp:2564
// TODO: implement inner merge
```

Currently, B-tree inner nodes are not merged when underflow occurs. This can lead to suboptimal tree height and wasted space.

#### Requirements:
- Detect underflow in inner nodes after deletion
- Merge with sibling or redistribute keys
- Update parent pointers correctly
- Handle root node special case (tree height reduction)

---

### 5. Page ID Reuse After Deletion

**Status**: Not Implemented  
**Complexity**: Medium  
**Files**: `src/calico.cpp`, `src/collection.cpp`

```cpp
// XXX at src/calico.cpp:2924
// XXX: should reuse page Id

// TODO at src/collection.cpp:1859
// TODO: Add to free list
```

#### Requirements:
- Maintain a free list of deallocated page IDs
- Integrate with PIDAllocator for recycling
- Handle crash recovery of free list state
- Consider per-index vs global free lists

---

### 6. Collection Page Compaction

**Status**: Not Implemented  
**Complexity**: Medium  
**Files**: `src/collection.cpp`

```cpp
// TODO at src/collection.cpp:2194-2195
// TODO: Free overflow pages if any
// TODO: Compact page if fragmentation is high
```

#### Requirements:
- Track fragmentation percentage per page
- Implement in-place compaction without blocking reads
- Reclaim overflow pages when documents are deleted
- Consider background compaction thread

---

### 7. Query Optimizer with Statistics

**Status**: Not Implemented  
**Complexity**: Medium  
**Files**: `src/collection.cpp`

```cpp
// TODO at src/collection.cpp:2313
// TODO: Use B-tree indices for indexed fields

// TODO at src/collection.cpp:2374  
// TODO: Use statistics and histogram
```

#### Requirements:
- Collect column statistics (min, max, distinct count, null count)
- Build histograms for selectivity estimation
- Choose between index scan vs full scan based on selectivity
- Support composite index selection

---

## Source Code TODOs

### Critical TODOs

| File | Line | Description | Priority |
|------|------|-------------|----------|
| `src/calico.cpp` | 2564 | Implement inner node merge in B-tree | Medium |
| `src/calico.cpp` | 2924 | Reuse page IDs after deletion | Medium |
| `src/collection.cpp` | 1859 | Add freed pages to free list | Medium |
| `src/collection.cpp` | 2194-2195 | Free overflow pages and compact | Medium |
| `src/collection.cpp` | 2313 | Use B-tree indices for filters | Medium |
| `src/collection.cpp` | 2374 | Implement statistics/histograms | Low |
| `src/ivfpq.cpp` | 1639 | Trigger online retraining | Medium |
| `include/caliby/ivfpq.hpp` | 22 | Implement actual inner product | High |
| `src/distance_diskann.cpp` | 1 | Entire file is TODO | Low |

### Optimization TODOs (XXX markers)

| File | Line | Description |
|------|------|-------------|
| `include/caliby/calico.hpp` | 1379 | Optimize GuardS lock acquisition |
| `include/caliby/calico.hpp` | 1776, 1797 | Optimize B-tree scan loop |
| `include/caliby/calico.hpp` | 1837 | Remove hack in scan implementation |

---

## Good First Issues

These are simpler tasks suitable for first-time contributors:

### 1. Add Logging to Remaining Files
- **Files**: Any file still using `std::cout`/`std::cerr` directly
- **Task**: Replace with `CALIBY_LOG_*` macros
- **Difficulty**: Easy

### 2. Add Unit Tests for Edge Cases
- **Files**: `tests/`
- **Task**: Add tests for boundary conditions (empty index, single element, max capacity)
- **Difficulty**: Easy

### 3. Documentation Improvements
- **Task**: Add docstrings to public APIs in header files
- **Difficulty**: Easy

### 4. Python Type Stubs
- **Task**: Create `.pyi` stub files for better IDE support
- **Difficulty**: Easy-Medium

### 5. Benchmark Scripts
- **Task**: Add comparison benchmarks against other vector databases
- **Difficulty**: Easy-Medium

---

## Performance Improvements

### 1. SIMD Optimizations
- Implement AVX-512 code paths (currently AVX2)
- ARM NEON support for Apple Silicon
- Auto-detection of CPU features

### 2. Memory Layout Optimizations
- Investigate cache-line alignment for hot data
- NUMA-aware allocation for multi-socket systems
- Huge page support for large indexes

### 3. I/O Optimizations
- io_uring support for async I/O (partially implemented)
- Prefetching improvements for sequential scans
- Compression for cold pages

### 4. Concurrency Improvements
- Lock-free data structures where possible
- Better work stealing in thread pools
- Reduce lock contention in hot paths

---

## How to Contribute

1. **Pick an Issue**: Choose a TODO from this list or create a new issue
2. **Discuss**: Comment on the issue to discuss your approach
3. **Fork & Branch**: Create a feature branch from `main`
4. **Implement**: Write code with tests and documentation
5. **Test**: Run `bash run_tests.sh` and ensure all tests pass
6. **PR**: Submit a pull request with a clear description

### Code Style
- Follow existing code style (4-space indentation, etc.)
- Use the logging system (`CALIBY_LOG_*` macros)
- Add appropriate log levels (DEBUG for verbose, INFO for important events)
- Write unit tests for new functionality

### Testing
```bash
# Run all tests
bash run_tests.sh

# Run specific test file
python -m pytest tests/test_specific.py -v
```

---

## Contact

- Open an issue for questions
- Join discussions in GitHub Discussions
- Tag maintainers for review: @zxjcarrot
