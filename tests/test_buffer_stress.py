#!/usr/bin/env python3
"""
Buffer Pool Stress and Eviction Behavior Tests

Tests the buffer pool under memory pressure, eviction, and
larger-than-memory scenarios.

Tests that need custom buffer configs run in subprocesses to avoid
conflicting with the shared session configuration.

Run with: pytest tests/test_buffer_stress.py -v
"""

import pytest
import os
import sys
import tempfile
import numpy as np
import time
import subprocess
import caliby

build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def _run_script(script, timeout=10):
    """Run a test script in a subprocess."""
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": f"{build_path}:{os.environ.get('PYTHONPATH', '')}"})
    return result


@pytest.fixture
def db_path():
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "test_buffer.db")
    yield path
    try:
        caliby.close()
    except Exception:
        pass
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBufferPoolConfiguration:
    """Test buffer pool configuration via subprocess (since set_buffer_config
    must be called before any system initialization)."""

    def test_small_buffer_large_index(self, db_path):
        script = f"""
import sys, os
sys.path.insert(0, '{build_path}')
import caliby, numpy as np, tempfile

with tempfile.TemporaryDirectory() as td:
    caliby.set_buffer_config(size_gb=0.05)
    caliby.open(td)
    dim, n = 64, 3000
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    idx = caliby.HnswIndex(max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
    idx.add_points(vectors, num_threads=2)
    q = vectors[0].copy()
    labels, _ = idx.search_knn(q, k=10, ef_search=50)
    assert len(labels) == 10
    assert labels[0] == 0
    print("PASS")
"""
        r = _run_script(script)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_default_buffer_config(self, db_path):
        script = f"""
import sys, os
sys.path.insert(0, '{build_path}')
import caliby, numpy as np, tempfile

with tempfile.TemporaryDirectory() as td:
    caliby.set_buffer_config(size_gb=0.5)
    caliby.open(td)
    dim, n = 32, 1000
    vectors = np.random.randn(n, dim).astype(np.float32)
    idx = caliby.HnswIndex(max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
    idx.add_points(vectors)
    labels, _ = idx.search_knn(vectors[0], k=10, ef_search=50)
    assert len(labels) == 10
    print("PASS")
"""
        r = _run_script(script)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_large_virtual_buffer(self, db_path):
        script = f"""
import sys, os
sys.path.insert(0, '{build_path}')
import caliby, numpy as np, tempfile

with tempfile.TemporaryDirectory() as td:
    caliby.set_buffer_config(size_gb=0.1, virtgb=10.0)
    caliby.open(td)
    dim, n = 32, 500
    vectors = np.random.randn(n, dim).astype(np.float32)
    idx = caliby.HnswIndex(max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
    idx.add_points(vectors)
    labels, _ = idx.search_knn(vectors[0], k=5, ef_search=50)
    assert len(labels) == 5
    print("PASS")
"""
        r = _run_script(script)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestForceEviction:
    """Test force eviction behavior via subprocess."""

    def test_force_evict_no_crash(self):
        script = f"""
import sys, os
sys.path.insert(0, '{build_path}')
import caliby, numpy as np, tempfile

with tempfile.TemporaryDirectory() as td:
    caliby.set_buffer_config(size_gb=0.2)
    caliby.open(td)
    dim, n = 64, 2000
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    idx = caliby.HnswIndex(max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
    idx.add_points(vectors)
    for portion in [0.1, 0.25, 0.5, 0.75]:
        caliby.force_evict_buffer_portion(portion)
    labels, _ = idx.search_knn(vectors[0].copy(), k=10, ef_search=100)
    assert len(labels) == 10
    assert labels[0] == 0
    print("PASS")
"""
        r = _run_script(script)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_force_evict_then_search(self):
        script = f"""
import sys, os
sys.path.insert(0, '{build_path}')
import caliby, numpy as np, tempfile

with tempfile.TemporaryDirectory() as td:
    caliby.set_buffer_config(size_gb=0.1)
    caliby.open(td)
    dim, n = 32, 3000
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    idx = caliby.HnswIndex(max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
    idx.add_points(vectors)
    caliby.force_evict_buffer_portion(0.9)
    for i in range(10):
        labels, _ = idx.search_knn(vectors[i].copy(), k=10, ef_search=100)
        assert len(labels) == 10
    print("PASS")
"""
        r = _run_script(script)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout

    def test_ivfpq_with_eviction(self):
        script = f"""
import sys, os
sys.path.insert(0, '{build_path}')
import caliby, numpy as np, tempfile

with tempfile.TemporaryDirectory() as td:
    caliby.set_buffer_config(size_gb=0.1)
    caliby.open(td)
    dim, n = 32, 2000
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)
    idx = caliby.IVFPQIndex(max_elements=n+500, dim=dim, num_clusters=16, num_subquantizers=4, skip_recovery=True, index_id=100)
    idx.train(vectors[:500])
    idx.add_points(vectors)
    caliby.force_evict_buffer_portion(0.8)
    labels, _ = idx.search_knn(vectors[0].copy(), k=10, nprobe=8)
    assert len(labels) == 10
    print("PASS")
"""
        r = _run_script(script)
        assert r.returncode == 0, r.stderr
        assert "PASS" in r.stdout


class TestIndexFlush:
    """Test explicit flush behavior using session config."""

    def test_flush_after_add(self, caliby_module, temp_dir):
        dim, n = 32, 1000
        vectors = np.random.randn(n, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        index = caliby_module.HnswIndex(
            max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        index.add_points(vectors)
        index.flush()
        caliby_module.flush_storage()
        del index

        index2 = caliby_module.HnswIndex(
            max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=False)
        assert index2.was_recovered()
        labels, _ = index2.search_knn(vectors[0].copy(), k=10, ef_search=100)
        assert labels[0] == 0

    def test_multiple_flush_cycles(self, caliby_module, temp_dir):
        dim, n = 32, 500
        vectors1 = np.random.randn(250, dim).astype(np.float32)
        vectors2 = np.random.randn(250, dim).astype(np.float32)

        index = caliby_module.HnswIndex(
            max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True, name="flush_cycles")
        index.add_points(vectors1)
        index.flush()
        caliby_module.flush_storage()
        index.add_points(vectors2)
        index.flush()
        caliby_module.flush_storage()
        del index

        index2 = caliby_module.HnswIndex(
            max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=False, name="flush_cycles")
        assert index2.was_recovered()


class TestCollectionBufferInteraction:
    """Test collection operations with buffer pool (session config)."""

    def test_collection_with_small_buffer(self, caliby_module, temp_dir):
        try:
            caliby_module.close()
        except Exception:
            pass
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("count", caliby.FieldType.INT)
        col = caliby.Collection("small_buf2", schema, vector_dim=32)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")

        n = 500
        contents = [f"Document about topic {i % 10}" for i in range(n)]
        metadatas = [{"category": f"cat_{i % 5}", "count": i} for i in range(n)]
        vectors = np.random.randn(n, 32).astype(np.float32).tolist()
        ids = col.add(contents, metadatas, vectors)
        assert len(ids) == n

        results = col.search_vector([0.1] * 32, "vec_idx", k=10)
        assert len(results) == 10
        text_results = col.search_text("topic", "text_idx", k=20)
        assert len(text_results) >= 1

    def test_collection_force_evict(self, caliby_module, temp_dir):
        try:
            caliby_module.close()
        except Exception:
            pass
        caliby_module.open(temp_dir)
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("evict_col2", schema, vector_dim=64)
        col.create_hnsw_index("vec_idx")

        n = 300
        vectors = np.random.randn(n, 64).astype(np.float32).tolist()
        ids = col.add(
            [f"doc {i}" for i in range(n)],
            [{"idx": i} for i in range(n)],
            vectors,
        )
        caliby.force_evict_buffer_portion(0.5)
        query = vectors[0]
        results = col.search_vector(query, "vec_idx", k=5)
        assert len(results) == 5


class TestBufferPerformance:
    """Basic performance under buffer pressure (session config)."""

    def test_search_performance_under_eviction(self, caliby_module, temp_dir):
        dim, n = 64, 2000
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)

        index = caliby_module.HnswIndex(
            max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        index.add_points(vectors)
        caliby.force_evict_buffer_portion(0.9)

        queries = vectors[:50]
        start = time.time()
        for q in queries:
            index.search_knn(q, k=10, ef_search=50)
        elapsed = time.time() - start
        qps = 50 / elapsed
        assert qps > 10, f"Search too slow after eviction: {qps:.1f} QPS"

    def test_add_performance(self, caliby_module, temp_dir):
        dim, n = 32, 2000
        vectors = np.random.randn(n, dim).astype(np.float32)
        index = caliby_module.HnswIndex(
            max_elements=n, dim=dim, M=16, ef_construction=100, skip_recovery=True)
        start = time.time()
        index.add_points(vectors, num_threads=4)
        elapsed = time.time() - start
        assert elapsed < 30, f"Adding {n} vectors took {elapsed:.1f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
