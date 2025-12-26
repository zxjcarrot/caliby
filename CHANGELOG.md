# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Caliby vector search library
- HNSW index with larger-than-memory support
- Multi-index support with unique index IDs for isolation
- Index naming feature with `name` parameter and `get_name()` method
- DiskANN index implementation
- IVF index with multiple quantizers (Flat, PQ, SQ)
- Python bindings with NumPy integration
- Unified buffer pool for all index types
- Persistence and crash recovery
- Concurrent search support
- L2, Inner Product, and Cosine distance metrics
- Comprehensive documentation and examples
- 38 comprehensive multi-index tests including naming functionality

### Performance
- Array-based translation achieving mmap-level performance
- 2MB huge page support for reduced TLB overhead
- Group prefetch API for high memory-level parallelism
- Hole punching for efficient memory reclamation

### Fixed
- Multi-index capacity calculation for correct page allocation
- Multi-index PID handling in residentPtr and prefetchPages operations
- Index ID reuse issue causing test hangs (implemented unique ID generation)

## [0.1.0] - 2024-XX-XX

### Added
- First public release
- Core buffer pool implementation
- HNSW index
- Basic Python bindings
- Documentation and examples

[Unreleased]: https://github.com/zxjcarrot/caliby/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zxjcarrot/caliby/releases/tag/v0.1.0
