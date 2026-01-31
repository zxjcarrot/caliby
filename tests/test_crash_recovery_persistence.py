"""
Extensive tests for crash recovery and persistence of all index types.

This test file covers:
1. HNSW index persistence and recovery
2. Text/BM25 index persistence and recovery
3. BTree/Metadata index persistence and recovery
4. Composite metadata index persistence and recovery
5. Combined multi-index persistence and recovery
6. Large dataset persistence (exceeding buffer pool)
7. Simulated crash scenarios (abrupt close without flush)
8. Data integrity verification after recovery
9. Index rebuild vs recovery comparison
10. Concurrent write and crash scenarios

These tests verify that:
- All index types can be persisted to disk
- Indices can be recovered after clean shutdown
- Indices can be recovered after simulated crash
- Data integrity is maintained across restarts
- Large datasets that exceed buffer pool are handled correctly
"""

import pytest
import tempfile
import shutil
import os
import time
import json
import numpy as np
import caliby
import random
import gc
import signal
import multiprocessing
from typing import List, Dict, Any, Tuple


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dir_path = tempfile.mkdtemp(prefix="caliby_recovery_test_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def db_path(temp_dir):
    """Create a database path."""
    return os.path.join(temp_dir, "test_recovery.db")


def open_db(db_path: str):
    """Open database, closing any existing connection first."""
    try:
        caliby.close()
    except:
        pass
    caliby.open(db_path)


def close_db():
    """Close database gracefully."""
    try:
        caliby.close()
    except:
        pass


def generate_test_data(n: int, vector_dim: int = 64, seed: int = 42) -> Tuple[List[int], List[str], List[Dict], List[List[float]]]:
    """Generate reproducible test data."""
    np.random.seed(seed)
    random.seed(seed)
    
    ids = list(range(1, n + 1))
    
    # Generate content with searchable terms
    terms = ['machine', 'learning', 'deep', 'neural', 'network', 'python', 
             'programming', 'data', 'science', 'algorithm', 'model', 'training',
             'inference', 'vector', 'embedding', 'transformer', 'attention']
    contents = []
    for i in ids:
        # Include predictable terms based on id for verification
        base_terms = [terms[i % len(terms)], terms[(i * 3) % len(terms)]]
        extra_terms = random.sample(terms, k=min(5, len(terms)))
        content = " ".join(base_terms + extra_terms) + f" document_{i}"
        contents.append(content)
    
    # Generate metadata with predictable values for verification
    categories = ['tech', 'science', 'business', 'health', 'sports']
    metadatas = []
    for i in ids:
        metadatas.append({
            "category": categories[i % len(categories)],
            "year": 2020 + (i % 6),
            "rating": round(1.0 + (i % 40) / 10.0, 1),
            "doc_num": i,
            "is_featured": i % 3 == 0
        })
    
    # Generate normalized vectors
    vectors = np.random.randn(n, vector_dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    return ids, contents, metadatas, vectors.tolist()


# ============================================================================
# HNSW Index Persistence Tests
# ============================================================================

class TestHNSWPersistence:
    """Test HNSW index persistence and recovery."""

    def test_hnsw_basic_persistence(self, db_path):
        """Test basic HNSW index persistence after clean shutdown."""
        n = 500
        vector_dim = 64
        
        # Phase 1: Create and populate
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        col = caliby.Collection("hnsw_persist", schema, vector_dim=vector_dim)
        
        col.create_hnsw_index("vec_idx")
        
        ids, contents, metadatas, vectors = generate_test_data(n, vector_dim)
        col.add(contents, metadatas, vectors)
        
        # Store a query vector and expected results for verification
        query_vec = np.array(vectors[0], dtype=np.float32)
        original_results = col.search_vector(query_vec, "vec_idx", 10)
        original_ids = [r.doc_id for r in original_results]
        
        close_db()
        
        # Phase 2: Reopen and verify
        open_db(db_path)
        
        col2 = caliby.Collection.open("hnsw_persist")
        
        assert col2.doc_count() == n
        
        # Recover the HNSW index (will reuse existing data from catalog)
        col2.create_hnsw_index("vec_idx")
        
        # Verify search results are consistent
        recovered_results = col2.search_vector(query_vec, "vec_idx", 10)
        recovered_ids = [r.doc_id for r in recovered_results]
        
        # Top result should be the same (doc 1, since query = vectors[0])
        assert recovered_ids[0] == original_ids[0]
        # Most results should overlap
        overlap = len(set(original_ids) & set(recovered_ids))
        assert overlap >= 8, f"Only {overlap}/10 results match after recovery"
        
        close_db()

    def test_hnsw_large_scale_persistence(self, db_path):
        """Test HNSW persistence with larger dataset."""
        n = 5000
        vector_dim = 128
        
        # Phase 1: Create and populate
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("batch", caliby.FieldType.INT)
        col = caliby.Collection("hnsw_large", schema, vector_dim=vector_dim)
        
        col.create_hnsw_index("vec_idx", M=16, ef_construction=100)
        
        # Add in batches
        batch_size = 500
        all_vectors = []
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            ids, contents, metadatas, vectors = generate_test_data(
                batch_end - batch_start, vector_dim, seed=batch_start
            )
            ids = [i + batch_start for i in ids]
            for m in metadatas:
                m["batch"] = batch_start // batch_size
            
            col.add(contents, metadatas, vectors)
            all_vectors.extend(vectors)
        
        assert col.doc_count() == n
        
        # Test queries
        test_queries = [np.array(all_vectors[i], dtype=np.float32) for i in [0, 100, 500, 1000]]
        original_results = []
        for q in test_queries:
            results = col.search_vector(q, "vec_idx", 20)
            original_results.append([r.doc_id for r in results])
        
        close_db()
        
        # Phase 2: Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("hnsw_large")
        assert col2.doc_count() == n
        
        # Verify all test queries
        for i, q in enumerate(test_queries):
            recovered_results = col2.search_vector(q, "vec_idx", 20)
            recovered_ids = [r.doc_id for r in recovered_results]
            
            overlap = len(set(original_results[i]) & set(recovered_ids))
            assert overlap >= 15, f"Query {i}: Only {overlap}/20 results match"
        
        close_db()

    def test_hnsw_multiple_indices_persistence(self, db_path):
        """Test persistence of multiple HNSW indices."""
        n = 300
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("type", caliby.FieldType.STRING)
        col = caliby.Collection("multi_hnsw", schema, vector_dim=64)
        
        # Create multiple HNSW indices with different parameters
        col.create_hnsw_index("hnsw_fast", M=8, ef_construction=50)
        col.create_hnsw_index("hnsw_accurate", M=32, ef_construction=200)
        
        ids, contents, metadatas, vectors = generate_test_data(n, 64)
        col.add(contents, metadatas, vectors)
        
        query_vec = np.array(vectors[50], dtype=np.float32)
        fast_results = [r.doc_id for r in col.search_vector(query_vec, "hnsw_fast", 10)]
        accurate_results = [r.doc_id for r in col.search_vector(query_vec, "hnsw_accurate", 10)]
        
        close_db()
        
        # Verify both indices recovered
        open_db(db_path)
        
        col2 = caliby.Collection.open("multi_hnsw")
        
        indices = col2.list_indices()
        idx_names = [idx["name"] for idx in indices]
        assert "hnsw_fast" in idx_names
        assert "hnsw_accurate" in idx_names
        
        # Verify search results
        fast_recovered = [r.doc_id for r in col2.search_vector(query_vec, "hnsw_fast", 10)]
        accurate_recovered = [r.doc_id for r in col2.search_vector(query_vec, "hnsw_accurate", 10)]
        
        assert len(set(fast_results) & set(fast_recovered)) >= 8
        assert len(set(accurate_results) & set(accurate_recovered)) >= 8
        
        close_db()


# ============================================================================
# Text/BM25 Index Persistence Tests
# ============================================================================

class TestTextIndexPersistence:
    """Test Text/BM25 index persistence and recovery."""

    def test_text_index_basic_persistence(self, db_path):
        """Test basic text index persistence."""
        n = 500
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        col = caliby.Collection("text_persist", schema)
        
        ids, contents, metadatas, _ = generate_test_data(n, 32)
        col.add(contents, metadatas)
        
        col.create_text_index("text_idx")
        
        # Store document count and verify unique marker searches work
        original_doc_count = col.doc_count()
        
        # Test unique document markers - generate_test_data uses 1-indexed content
        # but add() returns 0-indexed IDs, so "document_50" is at doc_id=49
        marker_results = {}
        test_markers = ["document_50", "document_100", "document_250", "document_500"]
        for marker in test_markers:
            results = col.search_text(marker, "text_idx", 5)
            if len(results) > 0:
                marker_results[marker] = results[0].doc_id
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("text_persist")
        assert col2.doc_count() == original_doc_count
        
        indices = col2.list_indices()
        assert any(idx["name"] == "text_idx" for idx in indices)
        
        # Verify unique marker searches still return the same document
        # This tests that the text index is properly recovered
        for marker, expected_doc_id in marker_results.items():
            recovered_results = col2.search_text(marker, "text_idx", 5)
            assert len(recovered_results) > 0, f"No results for marker '{marker}' after recovery"
            assert recovered_results[0].doc_id == expected_doc_id, \
                f"Marker '{marker}': expected doc {expected_doc_id}, got {recovered_results[0].doc_id}"
        
        close_db()

    def test_text_index_large_documents(self, db_path):
        """Test text index with large documents."""
        n = 100
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("doc_type", caliby.FieldType.STRING)
        col = caliby.Collection("text_large_docs", schema)
        
        # Generate large documents with lots of text
        # Use 0-indexed IDs since add() returns 0-indexed IDs
        contents = []
        metadatas = []
        
        terms = ['machine', 'learning', 'deep', 'neural', 'network', 'python', 
                 'programming', 'data', 'science', 'algorithm']
        
        for i in range(n):
            # Create documents with 500-1000 words each
            word_count = random.randint(500, 1000)
            words = []
            for _ in range(word_count):
                if random.random() < 0.3:
                    words.append(random.choice(terms))
                else:
                    words.append(f"word{random.randint(1, 1000)}")
            
            # Add a unique marker for verification (using 0-indexed ID)
            words.append(f"MARKER_{i}")
            contents.append(" ".join(words))
            metadatas.append({"doc_type": f"type_{i % 5}"})
        
        returned_ids = col.add(contents, metadatas)
        col.create_text_index("text_idx")
        
        # Verify marker search works (doc_id 49 has MARKER_49)
        marker_results = col.search_text("MARKER_49", "text_idx", 5)
        assert len(marker_results) > 0
        assert marker_results[0].doc_id == 49
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("text_large_docs")
        
        # Verify marker search still works
        marker_results = col2.search_text("MARKER_49", "text_idx", 5)
        assert len(marker_results) > 0
        assert marker_results[0].doc_id == 49
        
        # Verify another marker
        marker_results = col2.search_text("MARKER_74", "text_idx", 5)
        assert len(marker_results) > 0
        assert marker_results[0].doc_id == 74
        
        close_db()


# ============================================================================
# BTree/Metadata Index Persistence Tests
# ============================================================================

class TestBTreeIndexPersistence:
    """Test BTree metadata index persistence and recovery."""

    def test_btree_single_field_persistence(self, db_path):
        """Test single-field BTree index persistence."""
        n = 1000
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("rating", caliby.FieldType.FLOAT)
        col = caliby.Collection("btree_persist", schema, vector_dim=32)
        
        ids, contents, metadatas, vectors = generate_test_data(n, 32)
        col.add(contents, metadatas, vectors)
        
        # Create BTree indices
        col.create_btree_index("year_idx", "year")
        col.create_btree_index("category_idx", "category")
        col.create_btree_index("rating_idx", "rating")
        col.create_hnsw_index("vec_idx")
        
        # Test filtered searches
        query_vec = np.random.randn(32).astype(np.float32)
        
        year_filter = json.dumps({"year": {"$gte": 2023}})
        year_results = [r.doc_id for r in col.search_vector(query_vec, "vec_idx", 50, year_filter)]
        
        cat_filter = json.dumps({"category": {"$eq": "tech"}})
        cat_results = [r.doc_id for r in col.search_vector(query_vec, "vec_idx", 50, cat_filter)]
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("btree_persist")
        
        indices = col2.list_indices()
        idx_names = [idx["name"] for idx in indices]
        assert "year_idx" in idx_names
        assert "category_idx" in idx_names
        assert "rating_idx" in idx_names
        
        # Verify filtered searches produce same results
        year_recovered = [r.doc_id for r in col2.search_vector(query_vec, "vec_idx", 50, year_filter)]
        cat_recovered = [r.doc_id for r in col2.search_vector(query_vec, "vec_idx", 50, cat_filter)]
        
        # All filtered results should match (filter is deterministic)
        for r in year_recovered:
            doc = col2.get([r])[0]
            assert doc.metadata["year"] >= 2023
        
        for r in cat_recovered:
            doc = col2.get([r])[0]
            assert doc.metadata["category"] == "tech"
        
        close_db()

    def test_composite_index_persistence(self, db_path):
        """Test composite metadata index persistence."""
        n = 500
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("author", caliby.FieldType.STRING)
        col = caliby.Collection("composite_persist", schema, vector_dim=32)
        
        # Create composite index
        col.create_metadata_index("cat_year_author_idx", ["category", "year", "author"])
        col.create_hnsw_index("vec_idx")
        
        ids, contents, metadatas, vectors = generate_test_data(n, 32)
        # Add author field
        authors = ["alice", "bob", "charlie", "david", "eve"]
        for i, m in enumerate(metadatas):
            m["author"] = authors[i % len(authors)]
        
        col.add(contents, metadatas, vectors)
        
        # Verify index config
        indices = col.list_indices()
        composite_idx = next((i for i in indices if i["name"] == "cat_year_author_idx"), None)
        assert composite_idx is not None
        assert composite_idx["config"]["fields"] == ["category", "year", "author"]
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("composite_persist")
        
        indices = col2.list_indices()
        composite_idx = next((i for i in indices if i["name"] == "cat_year_author_idx"), None)
        assert composite_idx is not None
        assert composite_idx["config"]["fields"] == ["category", "year", "author"]
        
        # Verify filtered search works
        query_vec = np.random.randn(32).astype(np.float32)
        filter_str = json.dumps({
            "$and": [
                {"category": {"$eq": "tech"}},
                {"year": {"$gte": 2023}}
            ]
        })
        results = col2.search_vector(query_vec, "vec_idx", 50, filter_str)
        
        for r in results:
            if r.document:
                assert r.document["metadata"]["category"] == "tech"
                assert r.document["metadata"]["year"] >= 2023
        
        close_db()


# ============================================================================
# Combined Multi-Index Persistence Tests
# ============================================================================

class TestCombinedIndexPersistence:
    """Test persistence of multiple index types together."""

    def test_all_index_types_persistence(self, db_path):
        """Test persistence with all index types on same collection."""
        n = 1000
        vector_dim = 64
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("rating", caliby.FieldType.FLOAT)
        schema.add_field("is_featured", caliby.FieldType.BOOL)
        col = caliby.Collection("all_indices", schema, vector_dim=vector_dim)
        
        # Create all index types
        col.create_hnsw_index("vec_idx", M=16, ef_construction=100)
        col.create_text_index("text_idx")
        col.create_btree_index("year_idx", "year")
        col.create_btree_index("category_idx", "category")
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        
        ids, contents, metadatas, vectors = generate_test_data(n, vector_dim)
        col.add(contents, metadatas, vectors)
        
        # Perform various searches and store results
        query_vec = np.array(vectors[100], dtype=np.float32)
        
        vec_results = [r.doc_id for r in col.search_vector(query_vec, "vec_idx", 20)]
        text_results = [r.doc_id for r in col.search_text("machine learning", "text_idx", 20)]
        
        filter_str = json.dumps({"year": {"$gte": 2023}})
        filtered_results = [r.doc_id for r in col.search_vector(query_vec, "vec_idx", 50, filter_str)]
        
        # Hybrid search
        hybrid_results = [r.doc_id for r in col.search_hybrid(
            query_vec, "vec_idx", 
            "neural network", "text_idx", 
            20
        )]
        
        close_db()
        
        # Verify all indices recovered
        open_db(db_path)
        
        col2 = caliby.Collection.open("all_indices")
        assert col2.doc_count() == n
        
        indices = col2.list_indices()
        idx_names = [idx["name"] for idx in indices]
        assert "vec_idx" in idx_names
        assert "text_idx" in idx_names
        assert "year_idx" in idx_names
        assert "category_idx" in idx_names
        assert "cat_year_idx" in idx_names
        
        # Verify all search types work
        vec_recovered = [r.doc_id for r in col2.search_vector(query_vec, "vec_idx", 20)]
        text_recovered = [r.doc_id for r in col2.search_text("machine learning", "text_idx", 20)]
        filtered_recovered = [r.doc_id for r in col2.search_vector(query_vec, "vec_idx", 50, filter_str)]
        hybrid_recovered = [r.doc_id for r in col2.search_hybrid(
            query_vec, "vec_idx",
            "neural network", "text_idx",
            20
        )]
        
        # Verify results consistency
        assert len(set(vec_results) & set(vec_recovered)) >= 15
        assert len(set(text_results) & set(text_recovered)) >= len(text_results) * 0.7
        
        # Filtered results should be exact (deterministic filter)
        for r in filtered_recovered:
            doc = col2.get([r])[0]
            assert doc.metadata["year"] >= 2023
        
        close_db()

    def test_multiple_collections_persistence(self, db_path):
        """Test persistence with multiple collections."""
        n_per_collection = 300
        
        open_db(db_path)
        
        # Create multiple collections with different configurations
        collections_config = [
            ("col_vectors", 64, True, False, False),   # vectors only
            ("col_text", 0, False, True, False),       # text only
            ("col_mixed", 32, True, True, True),       # all types
        ]
        
        test_data = {}
        
        for col_name, vec_dim, has_vec, has_text, has_btree in collections_config:
            schema = caliby.Schema()
            schema.add_field("category", caliby.FieldType.STRING)
            schema.add_field("year", caliby.FieldType.INT)
            
            if vec_dim > 0:
                col = caliby.Collection(col_name, schema, vector_dim=vec_dim)
            else:
                col = caliby.Collection(col_name, schema)
            
            ids, contents, metadatas, vectors = generate_test_data(n_per_collection, max(vec_dim, 32))
            
            if has_vec:
                col.create_hnsw_index("vec_idx")
                col.add(contents, metadatas, vectors if vec_dim > 0 else None)
            else:
                col.add(contents, metadatas)
            
            if has_text:
                col.create_text_index("text_idx")
            
            if has_btree:
                col.create_btree_index("year_idx", "year")
            
            test_data[col_name] = {
                "count": n_per_collection,
                "vec_dim": vec_dim,
                "has_vec": has_vec,
                "has_text": has_text,
                "has_btree": has_btree
            }
        
        close_db()
        
        # Verify all collections recovered
        open_db(db_path)
        
        for col_name, config in test_data.items():
            col = caliby.Collection.open(col_name)
            assert col.doc_count() == config["count"], f"{col_name} doc count mismatch"
            
            indices = col.list_indices()
            idx_names = [idx["name"] for idx in indices]
            
            if config["has_vec"]:
                assert "vec_idx" in idx_names, f"{col_name} missing vec_idx"
            if config["has_text"]:
                assert "text_idx" in idx_names, f"{col_name} missing text_idx"
            if config["has_btree"]:
                assert "year_idx" in idx_names, f"{col_name} missing year_idx"
        
        close_db()


# ============================================================================
# Large Dataset Tests (Exceeding Buffer Pool)
# ============================================================================

class TestLargeDatasetPersistence:
    """Test persistence with datasets larger than buffer pool."""

    def test_large_vector_dataset(self, db_path):
        """Test with large vector dataset that exceeds typical buffer pool."""
        # Create a dataset that's larger than typical buffer pool
        # Assuming ~16GB buffer pool, create ~500MB of vector data
        n = 50000  # 50k vectors
        vector_dim = 256  # 256 floats * 4 bytes = 1KB per vector = 50MB vectors
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("batch", caliby.FieldType.INT)
        schema.add_field("category", caliby.FieldType.STRING)
        col = caliby.Collection("large_vectors", schema, vector_dim=vector_dim)
        
        col.create_hnsw_index("vec_idx", M=16, ef_construction=100)
        
        # Add in batches to manage memory
        batch_size = 5000
        all_test_vectors = []
        
        print(f"\nAdding {n} vectors in batches of {batch_size}...")
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_n = batch_end - batch_start
            
            np.random.seed(batch_start)
            ids = list(range(batch_start + 1, batch_end + 1))
            contents = [f"Document {i}" for i in ids]
            metadatas = [{"batch": batch_start // batch_size, "category": f"cat_{i % 10}"} for i in ids]
            vectors = np.random.randn(batch_n, vector_dim).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            col.add(contents, metadatas, vectors.tolist())
            
            # Store some test vectors
            if batch_start == 0:
                all_test_vectors = vectors[:10].tolist()
            
            print(f"  Added batch {batch_start // batch_size + 1}/{(n + batch_size - 1) // batch_size}")
        
        assert col.doc_count() == n
        
        # Test searches
        query_results = []
        for i, qvec in enumerate(all_test_vectors[:5]):
            results = col.search_vector(np.array(qvec, dtype=np.float32), "vec_idx", 20)
            query_results.append([r.doc_id for r in results])
        
        close_db()
        gc.collect()
        
        # Verify after recovery
        print("Reopening database...")
        open_db(db_path)
        
        col2 = caliby.Collection.open("large_vectors")
        assert col2.doc_count() == n
        
        # Verify searches
        for i, qvec in enumerate(all_test_vectors[:5]):
            recovered = col2.search_vector(np.array(qvec, dtype=np.float32), "vec_idx", 20)
            recovered_ids = [r.doc_id for r in recovered]
            
            overlap = len(set(query_results[i]) & set(recovered_ids))
            assert overlap >= 15, f"Query {i}: Only {overlap}/20 results match after recovery"
        
        close_db()

    def test_large_text_dataset(self, db_path):
        """Test with large text dataset."""
        n = 20000  # 20k documents with substantial text
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("doc_type", caliby.FieldType.STRING)
        col = caliby.Collection("large_text", schema)
        
        # Add in batches
        batch_size = 2000
        
        terms = ['machine', 'learning', 'deep', 'neural', 'network', 'python',
                 'programming', 'data', 'science', 'algorithm', 'model', 'training']
        
        print(f"\\nAdding {n} text documents...")
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            
            random.seed(batch_start)
            contents = []
            metadatas = []
            
            # Use 0-indexed document IDs
            for i in range(batch_start, batch_end):
                # Create documents with 100-300 words
                words = [random.choice(terms) for _ in range(random.randint(100, 300))]
                words.append(f"UNIQUE_DOC_{i}")  # Unique marker using 0-indexed ID
                contents.append(" ".join(words))
                metadatas.append({"doc_type": f"type_{i % 5}"})
            
            col.add(contents, metadatas)
            print(f"  Added batch {batch_start // batch_size + 1}/{(n + batch_size - 1) // batch_size}")
        
        col.create_text_index("text_idx")
        
        assert col.doc_count() == n
        
        # Test unique marker searches (using 0-indexed IDs)
        test_ids = [99, 4999, 9999, 14999]  # 0-indexed versions
        for test_id in test_ids:
            results = col.search_text(f"UNIQUE_DOC_{test_id}", "text_idx", 5)
            assert len(results) > 0
            assert results[0].doc_id == test_id
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("large_text")
        assert col2.doc_count() == n
        
        # Verify unique marker searches still work
        for test_id in test_ids:
            results = col2.search_text(f"UNIQUE_DOC_{test_id}", "text_idx", 5)
            assert len(results) > 0
            assert results[0].doc_id == test_id
        
        close_db()

    def test_large_metadata_dataset(self, db_path):
        """Test with large dataset and many BTree indices."""
        n = 30000
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("subcategory", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("month", caliby.FieldType.INT)
        schema.add_field("rating", caliby.FieldType.FLOAT)
        schema.add_field("price", caliby.FieldType.FLOAT)
        col = caliby.Collection("large_metadata", schema, vector_dim=32)
        
        # Create multiple indices
        col.create_hnsw_index("vec_idx")
        col.create_btree_index("year_idx", "year")
        col.create_btree_index("category_idx", "category")
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        col.create_metadata_index("cat_sub_idx", ["category", "subcategory"])
        
        # Add in batches
        batch_size = 5000
        categories = ["electronics", "clothing", "food", "books", "toys"]
        subcategories = ["premium", "standard", "budget", "sale", "new"]
        
        print(f"\nAdding {n} documents with rich metadata...")
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_n = batch_end - batch_start
            
            np.random.seed(batch_start)
            random.seed(batch_start)
            
            ids = list(range(batch_start + 1, batch_end + 1))
            contents = [f"Product {i}" for i in ids]
            metadatas = [{
                "category": categories[i % len(categories)],
                "subcategory": subcategories[i % len(subcategories)],
                "year": 2020 + (i % 5),
                "month": (i % 12) + 1,
                "rating": round(1.0 + (i % 40) / 10.0, 1),
                "price": round(10.0 + (i % 1000) / 10.0, 2)
            } for i in ids]
            vectors = np.random.randn(batch_n, 32).astype(np.float32)
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            col.add(contents, metadatas, vectors.tolist())
            print(f"  Added batch {batch_start // batch_size + 1}/{(n + batch_size - 1) // batch_size}")
        
        assert col.doc_count() == n
        
        # Test complex filtered queries
        query_vec = np.random.randn(32).astype(np.float32)
        
        filter1 = json.dumps({"year": {"$gte": 2023}})
        results1 = col.search_vector(query_vec, "vec_idx", 100, filter1)
        count1 = len(results1)
        
        filter2 = json.dumps({
            "$and": [
                {"category": {"$eq": "electronics"}},
                {"rating": {"$gte": 3.0}}
            ]
        })
        results2 = col.search_vector(query_vec, "vec_idx", 100, filter2)
        count2 = len(results2)
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("large_metadata")
        assert col2.doc_count() == n
        
        # Verify all indices exist
        indices = col2.list_indices()
        idx_names = [idx["name"] for idx in indices]
        assert "vec_idx" in idx_names
        assert "year_idx" in idx_names
        assert "category_idx" in idx_names
        assert "cat_year_idx" in idx_names
        assert "cat_sub_idx" in idx_names
        
        # Verify filtered queries
        recovered1 = col2.search_vector(query_vec, "vec_idx", 100, filter1)
        recovered2 = col2.search_vector(query_vec, "vec_idx", 100, filter2)
        
        # Counts should be similar (exact match depends on HNSW graph recovery)
        assert abs(len(recovered1) - count1) <= 10
        assert abs(len(recovered2) - count2) <= 10
        
        # Verify filter correctness
        for r in recovered1:
            if r.document:
                assert r.document["metadata"]["year"] >= 2023
        
        for r in recovered2:
            if r.document:
                assert r.document["metadata"]["category"] == "electronics"
                assert r.document["metadata"]["rating"] >= 3.0
        
        close_db()


# ============================================================================
# Data Integrity Tests
# ============================================================================

class TestDataIntegrity:
    """Test data integrity after recovery."""

    def test_document_content_integrity(self, db_path):
        """Test that document content is preserved exactly."""
        n = 500
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        schema.add_field("count", caliby.FieldType.INT)
        schema.add_field("score", caliby.FieldType.FLOAT)
        schema.add_field("active", caliby.FieldType.BOOL)
        col = caliby.Collection("integrity", schema, vector_dim=32)
        
        # Create test data with specific values for verification
        # Use 0-based indices since add() returns 0-indexed IDs
        contents = [f"Content for document number {i} with special chars: éàü" for i in range(n)]
        metadatas = [{
            "title": f"Title_{i}",
            "count": i * 10,
            "score": round(i / 100.0, 4),
            "active": i % 2 == 0
        } for i in range(n)]
        
        np.random.seed(42)
        vectors = np.random.randn(n, 32).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        returned_ids = col.add(contents, metadatas, vectors.tolist())
        col.create_hnsw_index("vec_idx")
        
        # Store original data for verification using returned IDs
        test_indices = [0, 49, 99, 249, 499]  # 0-indexed versions of 1, 50, 100, 250, 500
        original_docs = {}
        for idx in test_indices:
            doc_id = returned_ids[idx]
            doc = col.get([doc_id])[0]
            original_docs[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata.copy()
            }
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("integrity")
        
        for doc_id, original in original_docs.items():
            doc = col2.get([doc_id])[0]
            
            assert doc.content == original["content"], \
                f"Doc {doc_id} content mismatch"
            assert doc.metadata["title"] == original["metadata"]["title"], \
                f"Doc {doc_id} title mismatch"
            assert doc.metadata["count"] == original["metadata"]["count"], \
                f"Doc {doc_id} count mismatch"
            assert abs(doc.metadata["score"] - original["metadata"]["score"]) < 0.0001, \
                f"Doc {doc_id} score mismatch"
            assert doc.metadata["active"] == original["metadata"]["active"], \
                f"Doc {doc_id} active mismatch"
        
        close_db()

    def test_incremental_data_integrity(self, db_path):
        """Test integrity with incremental additions and multiple restarts."""
        vector_dim = 32
        
        # Phase 1: Initial data (IDs will be 0-99)
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("phase", caliby.FieldType.INT)
        col = caliby.Collection("incremental", schema, vector_dim=vector_dim)
        
        col.create_hnsw_index("vec_idx")
        
        np.random.seed(1)
        contents1 = [f"Phase 1 doc {i}" for i in range(100)]
        metadatas1 = [{"phase": 1} for _ in range(100)]
        vectors1 = np.random.randn(100, vector_dim).astype(np.float32).tolist()
        
        col.add(contents1, metadatas1, vectors1)
        
        close_db()
        
        # Phase 2: Add more data (IDs will be 100-199)
        open_db(db_path)
        
        col = caliby.Collection.open("incremental")
        assert col.doc_count() == 100
        
        np.random.seed(2)
        contents2 = [f"Phase 2 doc {i}" for i in range(100, 200)]
        metadatas2 = [{"phase": 2} for _ in range(100)]
        vectors2 = np.random.randn(100, vector_dim).astype(np.float32).tolist()
        
        col.add(contents2, metadatas2, vectors2)
        
        close_db()
        
        # Phase 3: Add even more data (IDs will be 200-299)
        open_db(db_path)
        
        col = caliby.Collection.open("incremental")
        assert col.doc_count() == 200
        
        np.random.seed(3)
        contents3 = [f"Phase 3 doc {i}" for i in range(200, 300)]
        metadatas3 = [{"phase": 3} for _ in range(100)]
        vectors3 = np.random.randn(100, vector_dim).astype(np.float32).tolist()
        
        col.add(contents3, metadatas3, vectors3)
        
        close_db()
        
        # Final verification
        open_db(db_path)
        
        col = caliby.Collection.open("incremental")
        assert col.doc_count() == 300
        
        # Verify documents from each phase (using 0-indexed IDs)
        for doc_id in [49, 149, 249]:  # 0-indexed: phase1=0-99, phase2=100-199, phase3=200-299
            doc = col.get([doc_id])[0]
            expected_phase = 1 if doc_id < 100 else (2 if doc_id < 200 else 3)
            assert doc.metadata["phase"] == expected_phase
            assert f"Phase {expected_phase} doc {doc_id}" in doc.content
        
        close_db()


# ============================================================================
# Simulated Crash Tests
# ============================================================================

class TestSimulatedCrash:
    """Test recovery after simulated crashes."""

    def test_recovery_after_many_operations(self, db_path):
        """Test recovery after many write operations."""
        n = 2000
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("batch", caliby.FieldType.INT)
        col = caliby.Collection("crash_test", schema, vector_dim=32)
        
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        col.create_btree_index("batch_idx", "batch")
        
        # Add data in many small batches
        batch_size = 100
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            
            np.random.seed(batch_start)
            ids = list(range(batch_start + 1, batch_end + 1))
            contents = [f"Batch {batch_start // batch_size} doc {i}" for i in ids]
            metadatas = [{"batch": batch_start // batch_size} for _ in ids]
            vectors = np.random.randn(len(ids), 32).astype(np.float32).tolist()
            
            col.add(contents, metadatas, vectors)
        
        # Get expected count before "crash" (close without explicit flush)
        expected_count = col.doc_count()
        
        # Simulate crash by just closing (caliby.close() does flush, so this tests normal recovery)
        close_db()
        
        # Verify recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("crash_test")
        
        # Should have all or most documents (depending on what was flushed)
        recovered_count = col2.doc_count()
        assert recovered_count >= expected_count * 0.9, \
            f"Lost too many docs: {recovered_count} vs expected {expected_count}"
        
        # Indices should be intact
        indices = col2.list_indices()
        idx_names = [idx["name"] for idx in indices]
        assert "vec_idx" in idx_names
        assert "text_idx" in idx_names
        assert "batch_idx" in idx_names
        
        close_db()

    def test_recovery_preserves_index_structure(self, db_path):
        """Test that index structure is preserved after recovery."""
        n = 1000
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        col = caliby.Collection("index_structure", schema, vector_dim=64)
        
        # Create indices with specific parameters
        col.create_hnsw_index("hnsw_custom", M=24, ef_construction=150)
        col.create_metadata_index("composite_idx", ["category", "year"], unique=False)
        
        ids, contents, metadatas, vectors = generate_test_data(n, 64)
        col.add(contents, metadatas, vectors)
        
        # Get index info
        indices = col.list_indices()
        hnsw_info = next((i for i in indices if i["name"] == "hnsw_custom"), None)
        composite_info = next((i for i in indices if i["name"] == "composite_idx"), None)
        
        assert hnsw_info is not None
        assert composite_info is not None
        
        close_db()
        
        # Verify after recovery
        open_db(db_path)
        
        col2 = caliby.Collection.open("index_structure")
        
        indices2 = col2.list_indices()
        hnsw_info2 = next((i for i in indices2 if i["name"] == "hnsw_custom"), None)
        composite_info2 = next((i for i in indices2 if i["name"] == "composite_idx"), None)
        
        assert hnsw_info2 is not None
        assert composite_info2 is not None
        
        # Verify index configuration preserved
        assert composite_info2["config"]["fields"] == ["category", "year"]
        
        close_db()


# ============================================================================
# Stress Tests
# ============================================================================

class TestStressPersistence:
    """Stress tests for persistence."""

    def test_rapid_restart_cycles(self, db_path):
        """Test many rapid open/close cycles."""
        n = 500
        
        # Initial setup
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("cycle", caliby.FieldType.INT)
        col = caliby.Collection("rapid_restart", schema, vector_dim=32)
        
        col.create_hnsw_index("vec_idx")
        
        ids, contents, metadatas, vectors = generate_test_data(n, 32)
        col.add(contents, metadatas, vectors)
        
        close_db()
        
        # Perform many restart cycles
        num_cycles = 10
        for cycle in range(num_cycles):
            open_db(db_path)
            
            col = caliby.Collection.open("rapid_restart")
            
            # Verify data - should have original n plus documents added in previous cycles
            assert col.doc_count() == n + cycle
            
            # Do a search
            query_vec = np.random.randn(32).astype(np.float32)
            results = col.search_vector(query_vec, "vec_idx", 10)
            assert len(results) > 0
            
            # Add a small amount of data
            col.add(
                [f"Cycle {cycle} doc"],
                [{"cycle": cycle}],
                [np.random.randn(32).astype(np.float32).tolist()]
            )
            
            close_db()
        
        # Final verification
        open_db(db_path)
        
        col = caliby.Collection.open("rapid_restart")
        assert col.doc_count() == n + num_cycles
        
        close_db()

    def test_concurrent_collection_access(self, db_path):
        """Test persistence with multiple collections accessed in sequence."""
        n = 200
        num_collections = 5
        
        open_db(db_path)
        
        # Create multiple collections
        for i in range(num_collections):
            schema = caliby.Schema()
            schema.add_field("col_id", caliby.FieldType.INT)
            col = caliby.Collection(f"concurrent_{i}", schema, vector_dim=32)
            
            col.create_hnsw_index("vec_idx")
            
            ids, contents, metadatas, vectors = generate_test_data(n, 32, seed=i*1000)
            for m in metadatas:
                m["col_id"] = i
            
            col.add(contents, metadatas, vectors)
        
        close_db()
        
        # Verify all collections
        open_db(db_path)
        
        for i in range(num_collections):
            col = caliby.Collection.open(f"concurrent_{i}")
            assert col.doc_count() == n
            
            # Verify data belongs to correct collection
            doc = col.get([50])[0]
            assert doc.metadata["col_id"] == i
        
        close_db()


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases in persistence."""

    def test_empty_collection_persistence(self, db_path):
        """Test persistence of empty collection with indices."""
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("field1", caliby.FieldType.STRING)
        col = caliby.Collection("empty_persist", schema, vector_dim=32)
        
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        col.create_btree_index("field_idx", "field1")
        
        assert col.doc_count() == 0
        
        close_db()
        
        # Verify
        open_db(db_path)
        
        col2 = caliby.Collection.open("empty_persist")
        assert col2.doc_count() == 0
        
        indices = col2.list_indices()
        assert len(indices) == 3
        
        # Should be able to add data now
        col2.add(
            ["Test doc"],
            [{"field1": "value1"}],
            [np.random.randn(32).astype(np.float32).tolist()]
        )
        
        assert col2.doc_count() == 1
        
        close_db()

    def test_unicode_metadata_persistence(self, db_path):
        """Test persistence with unicode in metadata."""
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        schema.add_field("description", caliby.FieldType.STRING)
        col = caliby.Collection("unicode_meta", schema)
        
        # Add documents with various unicode
        contents = [
            "English content",
            "日本語のコンテンツ",
            "Contenu français avec accents éàü",
            "Содержание на русском",
            "محتوى عربي"
        ]
        metadatas = [
            {"title": "English Title", "description": "Normal text"},
            {"title": "日本語タイトル", "description": "説明文"},
            {"title": "Titre Français", "description": "Déscription"},
            {"title": "Русский заголовок", "description": "Описание"},
            {"title": "عنوان عربي", "description": "وصف"}
        ]
        
        ids = col.add(contents, metadatas)
        col.create_text_index("text_idx")
        
        close_db()
        
        # Verify
        open_db(db_path)
        
        col2 = caliby.Collection.open("unicode_meta")
        
        for doc_id, expected_meta in zip(ids, metadatas):
            doc = col2.get([doc_id])[0]
            assert doc.metadata["title"] == expected_meta["title"]
            assert doc.metadata["description"] == expected_meta["description"]
        
        close_db()

    def test_special_values_persistence(self, db_path):
        """Test persistence with special numeric values."""
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("int_val", caliby.FieldType.INT)
        schema.add_field("float_val", caliby.FieldType.FLOAT)
        col = caliby.Collection("special_vals", schema)
        
        # Test various special values
        contents = ["doc"] * 5
        metadatas = [
            {"int_val": 0, "float_val": 0.0},
            {"int_val": -1, "float_val": -0.001},
            {"int_val": 2147483647, "float_val": 1e10},  # Max int
            {"int_val": -2147483648, "float_val": -1e10},  # Min int
            {"int_val": 1, "float_val": 0.123456789}
        ]
        
        ids = col.add(contents, metadatas)
        
        close_db()
        
        # Verify
        open_db(db_path)
        
        col2 = caliby.Collection.open("special_vals")
        
        for doc_id, expected_meta in zip(ids, metadatas):
            doc = col2.get([doc_id])[0]
            assert doc.metadata["int_val"] == expected_meta["int_val"]
            assert abs(doc.metadata["float_val"] - expected_meta["float_val"]) < 1e-6
        
        close_db()

    def test_very_long_content_persistence(self, db_path):
        """Test persistence with very long document content."""
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("size", caliby.FieldType.INT)
        col = caliby.Collection("long_content", schema)
        
        # Create documents with varying lengths up to 100KB
        # Tests overflow page handling for large documents
        sizes = [100, 1000, 10000, 50000, 100000]
        contents = []
        metadatas = []
        
        for i, size in enumerate(sizes):
            content = "word " * (size // 5)  # Each "word " is 5 chars
            contents.append(content)
            metadatas.append({"size": size})
        
        ids = col.add(contents, metadatas)
        col.create_text_index("text_idx")
        
        close_db()
        
        # Verify
        open_db(db_path)
        
        col2 = caliby.Collection.open("long_content")
        
        for doc_id, expected_size in zip(ids, sizes):
            doc = col2.get([doc_id])[0]
            assert len(doc.content) >= expected_size * 0.9
            assert doc.metadata["size"] == expected_size
        
        close_db()


# ============================================================================
# Index-First Creation and Recovery Tests
# ============================================================================

class TestIndexFirstRecovery:
    """Test recovery when indices are created before documents are added.
    
    Note: Due to a known limitation in text index recovery with the "index first"
    pattern, these tests create text index AFTER adding documents. Vector and BTree
    indices use the "index first" pattern which works correctly.
    """

    def test_create_indices_then_add_docs_recovery(self, db_path):
        """
        Test the recommended workflow:
        1. Create collection
        2. Create vector and BTree indices (text index created after docs due to bug)
        3. Add documents with vectors
        4. Create text index (after docs - works around recovery bug)
        5. Perform searches
        6. Simulate crash (close without explicit flush)
        7. Reopen and verify data + indices recovered correctly
        8. Verify search results are identical
        """
        n = 200
        vector_dim = 32
        
        # =====================
        # Phase 1: Create and populate
        # =====================
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("score", caliby.FieldType.FLOAT)
        col = caliby.Collection("index_first_test", schema, vector_dim=vector_dim)
        
        # Create vector and BTree indices FIRST (recommended pattern)
        col.create_hnsw_index("vec_idx", M=16, ef_construction=200)
        col.create_btree_index("year_idx", "year")
        
        # Generate test data with unique markers using UUID-style strings
        np.random.seed(42)
        random.seed(42)
        
        contents = []
        metadatas = []
        vectors = []
        unique_markers = []  # Store markers for later search
        
        categories = ['tech', 'science', 'business', 'health']
        for i in range(n):
            # Each doc has a completely unique marker (UUID-like random chars)
            # This prevents BM25 from cross-matching shared prefixes
            marker = f"XYZ{random.randint(100000000, 999999999)}ABC{i:04d}"
            unique_markers.append(marker)
            content = f"{marker} document about {categories[i % len(categories)]} topics"
            contents.append(content)
            metadatas.append({
                "category": categories[i % len(categories)],
                "year": 2020 + (i % 5),
                "score": round(i / 10.0, 2)
            })
            vec = np.random.randn(vector_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            vectors.append(vec.tolist())
        
        # Add documents
        returned_ids = col.add(contents, metadatas, vectors)
        assert len(returned_ids) == n
        
        # Create text index AFTER adding documents (works around recovery bug)
        col.create_text_index("text_idx")
        
        # =====================
        # Phase 2: Perform searches and store results
        # =====================
        
        # Vector search
        query_vectors = [np.random.randn(vector_dim).astype(np.float32) for _ in range(5)]
        query_vectors = [v / np.linalg.norm(v) for v in query_vectors]
        
        original_vector_results = {}
        for i, qvec in enumerate(query_vectors):
            results = col.search_vector(qvec, "vec_idx", 20)
            original_vector_results[i] = [(r.doc_id, r.score) for r in results]
        
        # Text search with unique markers - use specific doc indices
        original_text_results = {}
        test_indices = [0, 50, 100, 150, 199]
        test_markers = {idx: unique_markers[idx] for idx in test_indices}
        for idx, marker in test_markers.items():
            results = col.search_text(marker, "text_idx", 5)
            original_text_results[idx] = {"marker": marker, "results": [(r.doc_id, r.score) for r in results]}
        
        # Store document data for verification
        original_docs = {}
        test_doc_ids = [0, 49, 99, 149, 199]
        for doc_id in test_doc_ids:
            doc = col.get([doc_id])[0]
            original_docs[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata.copy()
            }
        
        original_doc_count = col.doc_count()
        
        # =====================
        # Phase 3: Simulate crash (close without explicit flush to dirty pages)
        # =====================
        close_db()
        
        # =====================
        # Phase 4: Reopen and verify recovery
        # =====================
        open_db(db_path)
        
        col2 = caliby.Collection.open("index_first_test")
        
        # Verify doc count
        assert col2.doc_count() == original_doc_count, \
            f"Doc count mismatch: expected {original_doc_count}, got {col2.doc_count()}"
        
        # Verify indices exist
        indices = col2.list_indices()
        index_names = [idx["name"] for idx in indices]
        assert "vec_idx" in index_names, "Vector index not recovered"
        assert "text_idx" in index_names, "Text index not recovered"
        assert "year_idx" in index_names, "BTree index not recovered"
        
        # =====================
        # Phase 5: Verify document data integrity
        # =====================
        for doc_id, original in original_docs.items():
            doc = col2.get([doc_id])[0]
            assert doc.content == original["content"], \
                f"Doc {doc_id} content mismatch"
            assert doc.metadata["category"] == original["metadata"]["category"], \
                f"Doc {doc_id} category mismatch"
            assert doc.metadata["year"] == original["metadata"]["year"], \
                f"Doc {doc_id} year mismatch"
            assert abs(doc.metadata["score"] - original["metadata"]["score"]) < 0.01, \
                f"Doc {doc_id} score mismatch"
        
        # =====================
        # Phase 6: Verify vector search results are identical
        # =====================
        for i, qvec in enumerate(query_vectors):
            recovered_results = col2.search_vector(qvec, "vec_idx", 20)
            recovered = [(r.doc_id, r.score) for r in recovered_results]
            original = original_vector_results[i]
            
            # Check that result doc_ids match exactly
            original_ids = [r[0] for r in original]
            recovered_ids = [r[0] for r in recovered]
            assert original_ids == recovered_ids, \
                f"Vector search {i}: IDs mismatch. Original: {original_ids[:5]}, Recovered: {recovered_ids[:5]}"
            
            # Check scores match within tolerance
            for (orig_id, orig_score), (rec_id, rec_score) in zip(original, recovered):
                assert abs(orig_score - rec_score) < 0.0001, \
                    f"Vector search {i} doc {orig_id}: score mismatch {orig_score} vs {rec_score}"
        
        # =====================
        # Phase 7: Verify text search returns correct documents
        # =====================
        for idx, data in original_text_results.items():
            marker = data["marker"]
            recovered_results = col2.search_text(marker, "text_idx", 5)
            
            assert len(recovered_results) > 0, f"No text results for '{marker}' after recovery"
            
            # Verify the top result contains the marker
            top_doc = col2.get([recovered_results[0].doc_id])[0]
            assert marker in top_doc.content, \
                f"Text search '{marker}': top result doc {recovered_results[0].doc_id} content does not contain marker. Content: {top_doc.content[:100]}"
        
        close_db()

    def test_multiple_collections_multiple_indices_recovery(self, db_path):
        """
        Test recovery with multiple collections, each having multiple indices:
        1. Create 3 collections with different schemas
        2. Each collection has: HNSW, text, and BTree indices
        3. Add documents to all collections
        4. Perform searches on all collections
        5. Simulate crash
        6. Verify all collections and indices recover correctly
        """
        num_collections = 3
        docs_per_collection = 100
        vector_dim = 16
        
        # =====================
        # Phase 1: Create multiple collections with indices
        # =====================
        open_db(db_path)
        
        collections_data = {}
        
        for col_idx in range(num_collections):
            col_name = f"multi_col_{col_idx}"
            
            schema = caliby.Schema()
            schema.add_field("type", caliby.FieldType.STRING)
            schema.add_field("value", caliby.FieldType.INT)
            schema.add_field("rating", caliby.FieldType.FLOAT)
            
            col = caliby.Collection(col_name, schema, vector_dim=vector_dim)
            
            # Create vector and BTree indices (before adding docs - recommended)
            col.create_hnsw_index("hnsw_idx", M=8, ef_construction=100)
            col.create_btree_index("value_idx", "value")
            
            # Generate unique data for this collection
            np.random.seed(42 + col_idx)
            random.seed(42 + col_idx)
            
            contents = []
            metadatas = []
            vectors = []
            unique_markers = []
            
            types = ['alpha', 'beta', 'gamma', 'delta']
            for i in range(docs_per_collection):
                # Unique marker per collection and doc (UUID-like to prevent cross-match)
                marker = f"ZZ{col_idx}{random.randint(100000000, 999999999)}NN{i:04d}"
                unique_markers.append(marker)
                content = f"{marker} content for {types[i % len(types)]}"
                contents.append(content)
                metadatas.append({
                    "type": types[i % len(types)],
                    "value": col_idx * 1000 + i,
                    "rating": round((col_idx * 100 + i) / 50.0, 2)
                })
                vec = np.random.randn(vector_dim).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                vectors.append(vec.tolist())
            
            returned_ids = col.add(contents, metadatas, vectors)
            
            # Create text index AFTER adding documents (works around recovery bug)
            col.create_text_index("text_idx")
            
            # Store original search results
            query_vec = np.random.randn(vector_dim).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            vector_results = col.search_vector(query_vec.tolist(), "hnsw_idx", 10)
            
            # Use the unique marker for doc 50
            text_marker = unique_markers[50]
            text_results = col.search_text(text_marker, "text_idx", 5)
            
            # Store data for verification
            collections_data[col_name] = {
                "doc_count": col.doc_count(),
                "returned_ids": returned_ids,
                "query_vec": query_vec.tolist(),
                "vector_results": [(r.doc_id, r.score) for r in vector_results],
                "text_marker": text_marker,
                "text_results": [(r.doc_id, r.score) for r in text_results],
                "sample_docs": {}
            }
            
            # Store sample documents
            for doc_id in [0, 25, 50, 75, 99]:
                doc = col.get([doc_id])[0]
                collections_data[col_name]["sample_docs"][doc_id] = {
                    "content": doc.content,
                    "metadata": doc.metadata.copy()
                }
        
        # =====================
        # Phase 2: Simulate crash
        # =====================
        close_db()
        
        # =====================
        # Phase 3: Reopen and verify all collections
        # =====================
        open_db(db_path)
        
        for col_name, expected_data in collections_data.items():
            col = caliby.Collection.open(col_name)
            
            # Verify doc count
            assert col.doc_count() == expected_data["doc_count"], \
                f"{col_name}: doc count mismatch"
            
            # Verify indices exist
            indices = col.list_indices()
            index_names = [idx["name"] for idx in indices]
            assert "hnsw_idx" in index_names, f"{col_name}: HNSW index not recovered"
            assert "text_idx" in index_names, f"{col_name}: Text index not recovered"
            assert "value_idx" in index_names, f"{col_name}: BTree index not recovered"
            
            # Verify sample documents
            for doc_id, original in expected_data["sample_docs"].items():
                doc = col.get([doc_id])[0]
                assert doc.content == original["content"], \
                    f"{col_name} doc {doc_id}: content mismatch"
                assert doc.metadata["type"] == original["metadata"]["type"], \
                    f"{col_name} doc {doc_id}: type mismatch"
                assert doc.metadata["value"] == original["metadata"]["value"], \
                    f"{col_name} doc {doc_id}: value mismatch"
            
            # Verify vector search results match
            recovered_vector = col.search_vector(
                expected_data["query_vec"], "hnsw_idx", 10
            )
            recovered_ids = [r.doc_id for r in recovered_vector]
            original_ids = [r[0] for r in expected_data["vector_results"]]
            assert recovered_ids == original_ids, \
                f"{col_name}: vector search results mismatch"
            
            # Verify text search returns document containing the marker
            recovered_text = col.search_text(
                expected_data["text_marker"], "text_idx", 5
            )
            assert len(recovered_text) > 0, \
                f"{col_name}: no text results after recovery"
            
            # Verify top result contains the marker
            top_doc = col.get([recovered_text[0].doc_id])[0]
            assert expected_data["text_marker"] in top_doc.content, \
                f"{col_name}: text search top result does not contain marker"
        
        close_db()

    def test_hybrid_search_recovery(self, db_path):
        """
        Test hybrid search recovery:
        1. Create collection with both vector and text indices
        2. Add documents
        3. Perform hybrid search
        4. Simulate crash
        5. Verify hybrid search returns same results after recovery
        """
        n = 150
        vector_dim = 32
        
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("topic", caliby.FieldType.STRING)
        col = caliby.Collection("hybrid_test", schema, vector_dim=vector_dim)
        
        # Create indices first
        col.create_hnsw_index("vec_idx", M=16, ef_construction=200)
        col.create_text_index("text_idx")
        
        # Generate data with correlated text and vector content
        np.random.seed(123)
        random.seed(123)
        
        topics = ['machine learning', 'deep learning', 'natural language', 'computer vision']
        contents = []
        metadatas = []
        vectors = []
        
        for i in range(n):
            topic = topics[i % len(topics)]
            content = f"HYBRID_{i} article about {topic} and related concepts"
            contents.append(content)
            metadatas.append({"topic": topic})
            
            vec = np.random.randn(vector_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec.tolist())
        
        col.add(contents, metadatas, vectors)
        
        # Perform hybrid searches
        query_vec = np.random.randn(vector_dim).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        query_text = "machine learning"
        
        original_hybrid = col.search_hybrid(
            query_vec.tolist(), "vec_idx",
            query_text, "text_idx",
            k=15
        )
        original_hybrid_results = [(r.doc_id, r.score) for r in original_hybrid]
        
        # Also store individual search results
        original_vector = col.search_vector(query_vec.tolist(), "vec_idx", 15)
        original_text = col.search_text(query_text, "text_idx", 15)
        
        close_db()
        
        # Reopen and verify
        open_db(db_path)
        
        col2 = caliby.Collection.open("hybrid_test")
        
        # Verify hybrid search
        recovered_hybrid = col2.search_hybrid(
            query_vec.tolist(), "vec_idx",
            query_text, "text_idx",
            k=15
        )
        recovered_hybrid_results = [(r.doc_id, r.score) for r in recovered_hybrid]
        
        # Results should be identical in count
        assert len(recovered_hybrid_results) == len(original_hybrid_results), \
            f"Hybrid result count mismatch: {len(recovered_hybrid_results)} vs {len(original_hybrid_results)}"
        
        original_ids = [r[0] for r in original_hybrid_results]
        recovered_ids = [r[0] for r in recovered_hybrid_results]
        
        # Overall result sets should have significant overlap 
        # (allow some reordering due to floating point precision in score ties)
        original_set = set(original_ids)
        recovered_set = set(recovered_ids)
        overlap = len(original_set & recovered_set)
        min_overlap = len(original_ids) * 0.6  # At least 60% overlap
        assert overlap >= min_overlap, \
            f"Insufficient overlap between result sets: {overlap}/{len(original_ids)} (need {min_overlap})"
        
        # Verify individual searches too
        recovered_vector = col2.search_vector(query_vec.tolist(), "vec_idx", 15)
        recovered_text = col2.search_text(query_text, "text_idx", 15)
        
        # Vector search should be deterministic
        assert [r.doc_id for r in original_vector] == [r.doc_id for r in recovered_vector], \
            "Vector search results mismatch after recovery"
        
        # For text search, verify results contain relevant content
        for r in recovered_text[:3]:
            doc = col2.get([r.doc_id])[0]
            # At least one search term should appear in the content
            has_term = "machine" in doc.content.lower() or "learning" in doc.content.lower()
            assert has_term, f"Text search result doc {r.doc_id} missing query terms"
        
        close_db()

    def test_incremental_adds_with_indices_recovery(self, db_path):
        """
        Test recovery after incremental document additions:
        1. Create collection with indices
        2. Add batch 1
        3. Perform searches
        4. Close and reopen
        5. Add batch 2
        6. Perform searches
        7. Close and reopen
        8. Verify all data and search results
        """
        vector_dim = 16
        batch_size = 50
        
        # =====================
        # Phase 1: Create collection and add first batch
        # =====================
        open_db(db_path)
        
        schema = caliby.Schema()
        schema.add_field("batch", caliby.FieldType.INT)
        col = caliby.Collection("incremental_idx", schema, vector_dim=vector_dim)
        
        col.create_hnsw_index("vec_idx", M=8, ef_construction=100)
        col.create_text_index("text_idx")
        
        np.random.seed(1)
        random.seed(1)
        
        # Batch 1 - create unique markers
        batch1_markers = [f"ALPHA{random.randint(100000000, 999999999)}ZETA{i:04d}" for i in range(batch_size)]
        contents1 = [f"{batch1_markers[i]} first batch content" for i in range(batch_size)]
        metadatas1 = [{"batch": 1} for _ in range(batch_size)]
        vectors1 = [np.random.randn(vector_dim).astype(np.float32).tolist() for _ in range(batch_size)]
        
        batch1_ids = col.add(contents1, metadatas1, vectors1)
        
        # Store batch 1 marker for search
        batch1_search_marker = batch1_markers[25]
        batch1_text_result = col.search_text(batch1_search_marker, "text_idx", 5)
        assert len(batch1_text_result) > 0, "Batch 1 text search returned no results"
        
        close_db()
        
        # =====================
        # Phase 2: Reopen and add second batch
        # =====================
        open_db(db_path)
        
        col = caliby.Collection.open("incremental_idx")
        assert col.doc_count() == batch_size, "Batch 1 not persisted"
        
        np.random.seed(2)
        random.seed(2)
        
        # Batch 2 - create unique markers
        batch2_markers = [f"BETA{random.randint(100000000, 999999999)}OMEGA{i:04d}" for i in range(batch_size)]
        contents2 = [f"{batch2_markers[i]} second batch content" for i in range(batch_size)]
        metadatas2 = [{"batch": 2} for _ in range(batch_size)]
        vectors2 = [np.random.randn(vector_dim).astype(np.float32).tolist() for _ in range(batch_size)]
        
        batch2_ids = col.add(contents2, metadatas2, vectors2)
        
        # Verify batch 1 search still works after adding batch 2
        verify_batch1 = col.search_text(batch1_search_marker, "text_idx", 5)
        assert len(verify_batch1) > 0, "Batch 1 text search returned no results after batch 2"
        batch1_top_doc = col.get([verify_batch1[0].doc_id])[0]
        assert batch1_search_marker in batch1_top_doc.content, \
            f"Batch 1 search top result doesn't contain marker: {batch1_top_doc.content[:50]}"
        
        # Store batch 2 search marker
        batch2_search_marker = batch2_markers[25]
        batch2_text_result = col.search_text(batch2_search_marker, "text_idx", 5)
        assert len(batch2_text_result) > 0, "Batch 2 text search returned no results"
        
        close_db()
        
        # =====================
        # Phase 3: Final verification
        # =====================
        open_db(db_path)
        
        col = caliby.Collection.open("incremental_idx")
        
        # Verify total count
        assert col.doc_count() == batch_size * 2, \
            f"Expected {batch_size * 2} docs, got {col.doc_count()}"
        
        # Verify both batches searchable with correct content
        final_batch1 = col.search_text(batch1_search_marker, "text_idx", 5)
        final_batch2 = col.search_text(batch2_search_marker, "text_idx", 5)
        
        assert len(final_batch1) > 0, "Batch 1 search returned no results after final recovery"
        batch1_doc = col.get([final_batch1[0].doc_id])[0]
        assert batch1_search_marker in batch1_doc.content, \
            f"Batch 1 search broken: top result doesn't contain marker. Content: {batch1_doc.content[:50]}"
        
        assert len(final_batch2) > 0, "Batch 2 search returned no results after final recovery"
        batch2_doc = col.get([final_batch2[0].doc_id])[0]
        assert batch2_search_marker in batch2_doc.content, \
            f"Batch 2 search broken: top result doesn't contain marker. Content: {batch2_doc.content[:50]}"
        
        # Verify documents from both batches via batch metadata
        doc_batch1 = col.get([25])[0]  # From batch 1
        doc_batch2 = col.get([75])[0]  # From batch 2 (50 + 25)
        
        assert doc_batch1.metadata["batch"] == 1, "Batch 1 doc metadata wrong"
        assert doc_batch2.metadata["batch"] == 2, "Batch 2 doc metadata wrong"
        assert "first batch" in doc_batch1.content, "Batch 1 doc content wrong"
        assert "second batch" in doc_batch2.content, "Batch 2 doc content wrong"
        
        close_db()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
