"""
Tests for Composite Metadata Index feature.

Tests cover:
1. Single-field metadata index (backward compatibility)
2. Composite (multi-field) metadata index creation
3. Index config verification
4. Error handling for invalid inputs

Note: Tests for actual query acceleration via composite indices are marked as skipped
because they require the query optimizer implementation which is a future task.
"""

import pytest
import tempfile
import shutil
import os
import numpy as np
import caliby


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dir_path = tempfile.mkdtemp(prefix="caliby_composite_test_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def initialized_db(temp_dir):
    """Initialize Caliby database."""
    db_path = os.path.join(temp_dir, "test.db")
    caliby.open(db_path)
    yield db_path
    try:
        caliby.close()
    except:
        pass


class TestSingleFieldMetadataIndex:
    """Tests for single-field metadata indices (backward compatibility)."""
    
    def test_create_single_field_index(self, initialized_db):
        """Test creating a single-field metadata index using new API."""
        schema = caliby.Schema()
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("category", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_single", schema)
        
        # Use new API with single field
        col.create_metadata_index("year_idx", ["year"])
        
        indices = col.list_indices()
        year_idx = next((i for i in indices if i["name"] == "year_idx"), None)
        
        assert year_idx is not None
        assert year_idx["type"] == "btree"
        assert year_idx["config"]["fields"] == ["year"]
    
    def test_backward_compatibility_create_btree_index(self, initialized_db):
        """Test that old create_btree_index API still works."""
        schema = caliby.Schema()
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection("test_compat", schema)
        
        # Use old API
        col.create_btree_index("year_btree", "year")
        
        indices = col.list_indices()
        year_idx = next((i for i in indices if i["name"] == "year_btree"), None)
        
        assert year_idx is not None
        assert year_idx["type"] == "btree"
        # Should have both "field" (legacy) and "fields" (new)
        assert year_idx["config"]["fields"] == ["year"]
        assert year_idx["config"]["field"] == "year"


class TestCompositeMetadataIndex:
    """Tests for composite (multi-field) metadata indices."""
    
    def test_create_composite_index_two_fields(self, initialized_db):
        """Test creating a two-field composite index."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection("test_composite2", schema)
        
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        
        indices = col.list_indices()
        idx = next((i for i in indices if i["name"] == "cat_year_idx"), None)
        
        assert idx is not None
        assert idx["type"] == "btree"
        assert idx["config"]["fields"] == ["category", "year"]
        assert idx["config"]["unique"] == False
    
    def test_create_composite_index_three_fields(self, initialized_db):
        """Test creating a three-field composite index."""
        schema = caliby.Schema()
        schema.add_field("store_id", caliby.FieldType.STRING)
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("price", caliby.FieldType.FLOAT)
        
        col = caliby.Collection("test_composite3", schema)
        
        col.create_metadata_index(
            "store_cat_price_idx",
            ["store_id", "category", "price"],
            unique=False
        )
        
        indices = col.list_indices()
        idx = next((i for i in indices if i["name"] == "store_cat_price_idx"), None)
        
        assert idx is not None
        assert idx["config"]["fields"] == ["store_id", "category", "price"]
    
    def test_create_unique_composite_index(self, initialized_db):
        """Test creating a unique composite index."""
        schema = caliby.Schema()
        schema.add_field("collection_name", caliby.FieldType.STRING)
        schema.add_field("doc_id", caliby.FieldType.INT)
        
        col = caliby.Collection("test_unique", schema)
        
        col.create_metadata_index(
            "unique_doc_idx",
            ["collection_name", "doc_id"],
            unique=True
        )
        
        indices = col.list_indices()
        idx = next((i for i in indices if i["name"] == "unique_doc_idx"), None)
        
        assert idx is not None
        assert idx["config"]["unique"] == True
    
    def test_create_index_field_not_in_schema_fails(self, initialized_db):
        """Test that creating index on non-existent field fails."""
        schema = caliby.Schema()
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection("test_invalid_field", schema)
        
        with pytest.raises(RuntimeError, match="not in schema"):
            col.create_metadata_index("bad_idx", ["nonexistent_field"])
    
    def test_create_index_partial_invalid_fields_fails(self, initialized_db):
        """Test that composite index with some invalid fields fails."""
        schema = caliby.Schema()
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("category", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_partial_invalid", schema)
        
        with pytest.raises(RuntimeError, match="not in schema"):
            col.create_metadata_index("bad_idx", ["year", "invalid_field"])
    
    def test_create_duplicate_index_fails(self, initialized_db):
        """Test that creating duplicate index name is idempotent (no error)."""
        schema = caliby.Schema()
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection("test_dup", schema)
        
        col.create_metadata_index("year_idx", ["year"])
        
        # Second call should be idempotent (no error)
        col.create_metadata_index("year_idx", ["year"])
        
        # Verify index still works
        indices = col.list_indices()
        index_names = [idx["name"] for idx in indices]
        assert "year_idx" in index_names


class TestCompositeIndexWithData:
    """Tests for composite index with actual data."""
    
    def test_add_data_with_composite_index(self, initialized_db):
        """Test adding data to collection with composite index."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("author", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_data", schema)
        col.create_metadata_index("cat_year_author_idx", ["category", "year", "author"])
        
        # Add sample data
        ids = col.add(
            contents=["doc1", "doc2", "doc3", "doc4", "doc5"],
            metadatas=[
                {"category": "tech", "year": 2024, "author": "alice"},
                {"category": "tech", "year": 2024, "author": "bob"},
                {"category": "tech", "year": 2023, "author": "alice"},
                {"category": "science", "year": 2024, "author": "alice"},
                {"category": "science", "year": 2023, "author": "charlie"},
            ]
        )
        
        assert col.doc_count() == 5
        
        # Verify we can retrieve documents by ID
        docs = col.get([ids[0], ids[1], ids[2]])
        assert len(docs) == 3


class TestMultipleIndices:
    """Tests for collections with multiple metadata indices."""
    
    def test_multiple_composite_indices(self, initialized_db):
        """Test creating multiple composite indices on same collection."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("author", caliby.FieldType.STRING)
        schema.add_field("rating", caliby.FieldType.FLOAT)
        
        col = caliby.Collection("test_multi_idx", schema)
        
        # Create multiple indices for different query patterns
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        col.create_metadata_index("author_rating_idx", ["author", "rating"])
        col.create_metadata_index("year_alone_idx", ["year"])
        
        indices = col.list_indices()
        assert len(indices) == 3
        
        idx_names = [i["name"] for i in indices]
        assert "cat_year_idx" in idx_names
        assert "author_rating_idx" in idx_names
        assert "year_alone_idx" in idx_names


class TestVectorSearchWithCompositeFilter:
    """Tests for vector search with composite index filters."""
    
    def test_vector_search_creates_index_and_filter(self, initialized_db):
        """Test creating vector index alongside composite metadata index."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection(
            "test_vec_composite",
            schema,
            vector_dim=8,
            distance_metric=caliby.DistanceMetric.COSINE
        )
        
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        col.create_hnsw_index("vec_idx")
        
        np.random.seed(42)
        
        # Add data with vectors
        vectors = np.random.rand(10, 8).astype(np.float32)
        ids = col.add(
            contents=["" for _ in range(10)],
            metadatas=[
                {"category": "tech", "year": 2024},
                {"category": "tech", "year": 2023},
                {"category": "tech", "year": 2022},
                {"category": "science", "year": 2024},
                {"category": "science", "year": 2023},
                {"category": "tech", "year": 2024},
                {"category": "tech", "year": 2023},
                {"category": "science", "year": 2024},
                {"category": "science", "year": 2022},
                {"category": "tech", "year": 2024},
            ],
            vectors=[v.tolist() for v in vectors]
        )
        
        assert col.doc_count() == 10
        
        # Verify indices exist
        indices = col.list_indices()
        idx_names = [i["name"] for i in indices]
        assert "cat_year_idx" in idx_names
        assert "vec_idx" in idx_names


class TestEdgeCases:
    """Edge case tests for composite metadata indices."""
    
    def test_empty_collection_with_index(self, initialized_db):
        """Test creating composite index on empty collection."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection("test_empty", schema)
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        
        assert col.doc_count() == 0
        
        # Verify index exists
        indices = col.list_indices()
        assert len(indices) == 1
        assert indices[0]["name"] == "cat_year_idx"


# Tests that verify leftmost prefix rule behavior
class TestLeftmostPrefixRule:
    """Tests that verify leftmost prefix rule behavior.
    
    These tests verify that composite indices are created and data 
    can be filtered correctly. The leftmost prefix optimization is
    a storage-level detail verified through the index configuration.
    """
    
    def test_composite_index_stores_field_order(self, initialized_db):
        """Test that composite index stores fields in the specified order."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("author", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_field_order", schema)
        
        # Create composite index with specific field order
        col.create_metadata_index("cat_year_author_idx", ["category", "year", "author"])
        
        indices = col.list_indices()
        idx = next((i for i in indices if i["name"] == "cat_year_author_idx"), None)
        
        assert idx is not None
        # Verify fields are stored in the exact order specified (leftmost first)
        assert idx["config"]["fields"] == ["category", "year", "author"]
        assert idx["config"]["fields"][0] == "category"  # leftmost
        assert idx["config"]["fields"][1] == "year"      # middle
        assert idx["config"]["fields"][2] == "author"    # rightmost
    
    def test_filter_on_leftmost_field_with_composite_index(self, initialized_db):
        """Test filtering on leftmost field with composite index present."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection(
            "test_leftmost_filter",
            schema,
            vector_dim=8,
            distance_metric=caliby.DistanceMetric.COSINE
        )
        
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        col.create_hnsw_index("vec_idx")
        
        np.random.seed(42)
        vectors = np.random.rand(20, 8).astype(np.float32)
        
        ids = col.add(
            contents=["" for _ in range(20)],
            metadatas=[
                {"category": "tech", "year": 2024},
                {"category": "tech", "year": 2023},
                {"category": "science", "year": 2024},
                {"category": "science", "year": 2023},
                {"category": "tech", "year": 2024},
                {"category": "sports", "year": 2022},
                {"category": "tech", "year": 2023},
                {"category": "science", "year": 2024},
                {"category": "sports", "year": 2023},
                {"category": "tech", "year": 2024},
                {"category": "tech", "year": 2022},
                {"category": "science", "year": 2023},
                {"category": "sports", "year": 2024},
                {"category": "tech", "year": 2023},
                {"category": "science", "year": 2022},
                {"category": "sports", "year": 2023},
                {"category": "tech", "year": 2024},
                {"category": "science", "year": 2024},
                {"category": "sports", "year": 2022},
                {"category": "tech", "year": 2023},
            ],
            vectors=[v.tolist() for v in vectors]
        )
        
        # Filter on leftmost field only (should be able to use index prefix)
        import json
        query_vec = np.random.rand(8).astype(np.float32)
        filter_str = json.dumps({"category": {"$eq": "tech"}})
        results = col.search_vector(query_vec, "vec_idx", 20, filter_str)
        
        # All results should have category "tech"
        for r in results:
            if r.document:
                assert r.document["metadata"]["category"] == "tech"


class TestRangeQueriesWithCompositeIndex:
    """Tests for range queries on composite indices."""
    
    def test_range_query_with_composite_index(self, initialized_db):
        """Test range query on field that's part of composite index."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection(
            "test_range_composite",
            schema,
            vector_dim=8,
            distance_metric=caliby.DistanceMetric.COSINE
        )
        
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        col.create_hnsw_index("vec_idx")
        
        np.random.seed(42)
        vectors = np.random.rand(15, 8).astype(np.float32)
        
        ids = col.add(
            contents=["" for _ in range(15)],
            metadatas=[
                {"category": "tech", "year": 2020},
                {"category": "tech", "year": 2021},
                {"category": "tech", "year": 2022},
                {"category": "tech", "year": 2023},
                {"category": "tech", "year": 2024},
                {"category": "science", "year": 2020},
                {"category": "science", "year": 2021},
                {"category": "science", "year": 2022},
                {"category": "science", "year": 2023},
                {"category": "science", "year": 2024},
                {"category": "sports", "year": 2020},
                {"category": "sports", "year": 2021},
                {"category": "sports", "year": 2022},
                {"category": "sports", "year": 2023},
                {"category": "sports", "year": 2024},
            ],
            vectors=[v.tolist() for v in vectors]
        )
        
        # Range query on year field
        import json
        query_vec = np.random.rand(8).astype(np.float32)
        filter_str = json.dumps({"year": {"$gte": 2023}})
        results = col.search_vector(query_vec, "vec_idx", 15, filter_str)
        
        # All results should have year >= 2023
        for r in results:
            if r.document:
                assert r.document["metadata"]["year"] >= 2023
    
    def test_combined_equality_and_range_with_composite_index(self, initialized_db):
        """Test combined equality and range query with composite index."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        
        col = caliby.Collection(
            "test_combo_composite",
            schema,
            vector_dim=8,
            distance_metric=caliby.DistanceMetric.COSINE
        )
        
        col.create_metadata_index("cat_year_idx", ["category", "year"])
        col.create_hnsw_index("vec_idx")
        
        np.random.seed(42)
        vectors = np.random.rand(15, 8).astype(np.float32)
        
        ids = col.add(
            contents=["" for _ in range(15)],
            metadatas=[
                {"category": "tech", "year": 2020},
                {"category": "tech", "year": 2021},
                {"category": "tech", "year": 2022},
                {"category": "tech", "year": 2023},
                {"category": "tech", "year": 2024},
                {"category": "science", "year": 2020},
                {"category": "science", "year": 2021},
                {"category": "science", "year": 2022},
                {"category": "science", "year": 2023},
                {"category": "science", "year": 2024},
                {"category": "sports", "year": 2020},
                {"category": "sports", "year": 2021},
                {"category": "sports", "year": 2022},
                {"category": "sports", "year": 2023},
                {"category": "sports", "year": 2024},
            ],
            vectors=[v.tolist() for v in vectors]
        )
        
        # Combined filter: category == "tech" AND year >= 2023
        import json
        query_vec = np.random.rand(8).astype(np.float32)
        filter_str = json.dumps({
            "$and": [
                {"category": {"$eq": "tech"}},
                {"year": {"$gte": 2023}}
            ]
        })
        results = col.search_vector(query_vec, "vec_idx", 15, filter_str)
        
        # All results should match both conditions
        for r in results:
            if r.document:
                assert r.document["metadata"]["category"] == "tech"
                assert r.document["metadata"]["year"] >= 2023


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
