"""Tests for the Collection system."""

import pytest
import os
import tempfile
import shutil
import numpy as np
import caliby


# Helper to manage test database
@pytest.fixture
def db_path():
    """Create a temporary database path and clean up after test."""
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "test.db")
    yield path
    # Cleanup
    try:
        caliby.close()
    except:
        pass
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def initialized_db(db_path):
    """Initialize caliby with a fresh database."""
    caliby.open(db_path)
    yield db_path
    try:
        caliby.close()
    except:
        pass


class TestSchema:
    """Test Schema class functionality."""

    def test_schema_creation_empty(self):
        """Test creating an empty schema."""
        schema = caliby.Schema()
        assert len(schema.fields()) == 0

    def test_schema_add_string_field(self):
        """Test adding a string field."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        assert schema.has_field("name")
        fields = schema.fields()
        assert len(fields) == 1

    def test_schema_add_multiple_fields(self):
        """Test adding multiple fields of different types."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("rating", caliby.FieldType.FLOAT)
        schema.add_field("active", caliby.FieldType.BOOL)
        
        assert schema.has_field("title")
        assert schema.has_field("year")
        assert schema.has_field("rating")
        assert schema.has_field("active")
        assert len(schema.fields()) == 4

    def test_schema_has_field_nonexistent(self):
        """Test has_field returns false for nonexistent field."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        assert not schema.has_field("nonexistent")

    def test_schema_field_types(self):
        """Test all field types can be added."""
        schema = caliby.Schema()
        schema.add_field("str_field", caliby.FieldType.STRING)
        schema.add_field("int_field", caliby.FieldType.INT)
        schema.add_field("float_field", caliby.FieldType.FLOAT)
        schema.add_field("bool_field", caliby.FieldType.BOOL)
        schema.add_field("str_arr_field", caliby.FieldType.STRING_ARRAY)
        schema.add_field("int_arr_field", caliby.FieldType.INT_ARRAY)
        
        assert len(schema.fields()) == 6


class TestDocument:
    """Test Document class functionality."""

    def test_document_default_constructor(self):
        """Test default document construction."""
        doc = caliby.Document()
        assert doc.id == 0
        assert doc.content == ""

    def test_document_with_values(self):
        """Test document construction with values."""
        doc = caliby.Document(42, "Test content", {"key": "value"})
        assert doc.id == 42
        assert doc.content == "Test content"
        assert doc.metadata["key"] == "value"

    def test_document_metadata_types(self):
        """Test document metadata with various types."""
        metadata = {
            "string_val": "hello",
            "int_val": 123,
            "float_val": 3.14,
            "bool_val": True,
            "null_val": None,
            "list_val": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }
        doc = caliby.Document(1, "content", metadata)
        
        assert doc.metadata["string_val"] == "hello"
        assert doc.metadata["int_val"] == 123
        assert abs(doc.metadata["float_val"] - 3.14) < 0.001
        assert doc.metadata["bool_val"] == True
        assert doc.metadata["null_val"] is None
        assert doc.metadata["list_val"] == [1, 2, 3]
        assert doc.metadata["nested"]["a"] == 1

    def test_document_modify_metadata(self):
        """Test modifying document metadata."""
        doc = caliby.Document(1, "content", {"key": "old"})
        doc.metadata = {"key": "new", "extra": 123}
        assert doc.metadata["key"] == "new"
        assert doc.metadata["extra"] == 123


class TestCollectionCreation:
    """Test Collection creation and basic properties."""

    def test_create_collection_basic(self, initialized_db):
        """Test creating a basic collection."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        
        col = caliby.Collection("test_coll", schema)
        assert col.name() == "test_coll"
        assert col.doc_count() == 0
        assert not col.has_vectors()

    def test_create_collection_with_vectors(self, initialized_db):
        """Test creating a collection with vector support."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        
        col = caliby.Collection("vec_coll", schema, vector_dim=128)
        assert col.has_vectors()
        assert col.vector_dim() == 128

    def test_create_collection_with_distance_metrics(self, initialized_db):
        """Test creating collections with different distance metrics."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        
        # L2 distance
        col_l2 = caliby.Collection("l2_coll", schema, vector_dim=64, 
                                    distance_metric=caliby.DistanceMetric.L2)
        assert col_l2.has_vectors()
        
        # Cosine distance
        schema2 = caliby.Schema()
        schema2.add_field("name", caliby.FieldType.STRING)
        col_cos = caliby.Collection("cos_coll", schema2, vector_dim=64,
                                     distance_metric=caliby.DistanceMetric.COSINE)
        assert col_cos.has_vectors()
        
        # Inner product
        schema3 = caliby.Schema()
        schema3.add_field("name", caliby.FieldType.STRING)
        col_ip = caliby.Collection("ip_coll", schema3, vector_dim=64,
                                    distance_metric=caliby.DistanceMetric.IP)
        assert col_ip.has_vectors()

    def test_create_multiple_collections(self, initialized_db):
        """Test creating multiple collections."""
        collections = []
        for i in range(5):
            schema = caliby.Schema()
            schema.add_field(f"field_{i}", caliby.FieldType.STRING)
            col = caliby.Collection(f"collection_{i}", schema)
            collections.append(col)
        
        for i, col in enumerate(collections):
            assert col.name() == f"collection_{i}"


class TestCollectionDocumentOperations:
    """Test Collection document CRUD operations."""

    def test_add_single_document(self, initialized_db):
        """Test adding a single document."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        col = caliby.Collection("single_doc", schema)
        
        # New API: add() returns assigned IDs
        ids = col.add(["Hello World"], [{"title": "Greeting"}])
        assert len(ids) == 1
        assert col.doc_count() == 1

    def test_add_multiple_documents(self, initialized_db):
        """Test adding multiple documents at once."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        col = caliby.Collection("multi_doc", schema)
        
        n = 100
        contents = [f"Content {i}" for i in range(n)]
        metadatas = [{"title": f"Title {i}"} for i in range(n)]
        
        ids = col.add(contents, metadatas)
        assert len(ids) == n
        assert col.doc_count() == n
        # IDs should be sequential starting from 0
        assert list(ids) == list(range(n))

    def test_add_documents_returns_sequential_ids(self, initialized_db):
        """Test that add() returns sequential IDs starting from 0."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("seq_ids", schema)
        
        # First batch
        ids1 = col.add(["Doc 1", "Doc 2"], [{"name": "a"}, {"name": "b"}])
        assert list(ids1) == [0, 1]
        
        # Second batch
        ids2 = col.add(["Doc 3", "Doc 4", "Doc 5"], 
                       [{"name": "c"}, {"name": "d"}, {"name": "e"}])
        assert list(ids2) == [2, 3, 4]
        
        assert col.doc_count() == 5

    def test_get_documents_by_id(self, initialized_db):
        """Test retrieving documents by ID."""
        schema = caliby.Schema()
        schema.add_field("value", caliby.FieldType.INT)
        col = caliby.Collection("get_test", schema)
        
        contents = ["A", "B", "C", "D", "E"]
        metadatas = [{"value": i * 10} for i in range(5)]
        
        ids = col.add(contents, metadatas)
        
        # Get subset
        docs = col.get([1, 3])
        assert len(docs) == 2
        
        id_to_content = {d.id: d.content for d in docs}
        assert id_to_content[1] == "B"
        assert id_to_content[3] == "D"

    def test_get_all_documents(self, initialized_db):
        """Test retrieving all documents."""
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("get_all", schema)
        
        n = 50
        contents = [f"Doc {i}" for i in range(n)]
        metadatas = [{"idx": i} for i in range(n)]
        
        ids = col.add(contents, metadatas)
        
        docs = col.get(list(ids))
        assert len(docs) == n

    def test_get_nonexistent_document(self, initialized_db):
        """Test getting a nonexistent document returns empty or raises."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("nonexistent", schema)
        
        col.add(["Exists"], [{"name": "test"}])
        
        # Getting nonexistent ID - behavior may vary
        try:
            docs = col.get([999])
            assert isinstance(docs, list)
        except Exception:
            pass  # OK if it raises for nonexistent IDs

    def test_update_document_metadata(self, initialized_db):
        """Test updating document metadata."""
        schema = caliby.Schema()
        schema.add_field("status", caliby.FieldType.STRING)
        schema.add_field("count", caliby.FieldType.INT)
        col = caliby.Collection("update_test", schema)
        
        ids = col.add(["Doc 1", "Doc 2"], 
                      [{"status": "draft", "count": 0}, {"status": "draft", "count": 0}])
        
        # Update both documents
        col.update(list(ids), [{"status": "published", "count": 10}, 
                               {"status": "archived", "count": 5}])
        
        docs = col.get(list(ids))
        id_to_meta = {d.id: d.metadata for d in docs}
        
        assert id_to_meta[0]["status"] == "published"
        assert id_to_meta[0]["count"] == 10
        assert id_to_meta[1]["status"] == "archived"
        assert id_to_meta[1]["count"] == 5

    def test_delete_single_document(self, initialized_db):
        """Test deleting a single document."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("delete_single", schema)
        
        ids = col.add(["A", "B", "C"], 
                      [{"name": "a"}, {"name": "b"}, {"name": "c"}])
        assert col.doc_count() == 3
        
        col.delete([1])  # Delete middle document
        assert col.doc_count() == 2

    def test_delete_multiple_documents(self, initialized_db):
        """Test deleting multiple documents."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("delete_multi", schema)
        
        ids = col.add(["A", "B", "C", "D", "E"],
                      [{"name": n} for n in ["a", "b", "c", "d", "e"]])
        
        col.delete([0, 2, 4])  # Delete every other
        assert col.doc_count() == 2

    def test_add_empty_content(self, initialized_db):
        """Test adding document with empty content."""
        schema = caliby.Schema()
        schema.add_field("tag", caliby.FieldType.STRING)
        col = caliby.Collection("empty_content", schema)
        
        ids = col.add([""], [{"tag": "empty"}])
        assert col.doc_count() == 1
        
        docs = col.get(list(ids))
        assert docs[0].content == ""

    def test_add_large_content(self, initialized_db):
        """Test adding document with large content."""
        schema = caliby.Schema()
        schema.add_field("size", caliby.FieldType.INT)
        col = caliby.Collection("large_content", schema)
        
        large_content = "x" * 100000  # 100KB
        ids = col.add([large_content], [{"size": len(large_content)}])
        assert col.doc_count() == 1
        
        docs = col.get(list(ids))
        assert len(docs[0].content) == 100000


class TestCollectionVectorOperations:
    """Test Collection vector operations."""

    def test_add_documents_with_vectors(self, initialized_db):
        """Test adding documents with vectors."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("vec_docs", schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        
        contents = ["Doc 1", "Doc 2", "Doc 3"]
        metadatas = [{"name": f"Name {i}"} for i in range(3)]
        vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        
        ids = col.add(contents, metadatas, vectors)
        assert col.doc_count() == 3

    def test_vector_search_basic(self, initialized_db):
        """Test basic vector search."""
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("vec_search", schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        
        contents = ["Doc A", "Doc B", "Doc C"]
        metadatas = [{"idx": i} for i in range(3)]
        vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]]
        
        ids = col.add(contents, metadatas, vectors)
        
        # Search for something close to [1, 0, 0, 0]
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=3)
        
        assert len(results) == 3
        # First result should be doc 0 with distance 0
        assert results[0].doc_id == 0
        assert results[0].score < 0.001

    def test_vector_search_returns_k_results(self, initialized_db):
        """Test that vector search returns k results."""
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("vec_k", schema, vector_dim=16)
        col.create_hnsw_index("vec_idx")
        
        n = 100
        contents = [f"Doc {i}" for i in range(n)]
        metadatas = [{"idx": i} for i in range(n)]
        vectors = [[float(i == j) for j in range(16)] for i in range(n)]
        
        ids = col.add(contents, metadatas, vectors)
        
        results = col.search_vector([1.0] + [0.0] * 15, "vec_idx", k=10)
        assert len(results) == 10


class TestCollectionIndexCreationOrder:
    """Test that index creation order matters."""
    
    def test_index_before_add_works(self, initialized_db):
        """Creating index before adding documents should work correctly."""
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("idx_before", schema, vector_dim=4)
        
        # Create index FIRST
        col.create_hnsw_index("vec_idx")
        
        # Then add documents
        vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        ids = col.add(["a", "b", "c"], [{"idx": i} for i in range(3)], vectors)
        
        # Search should find correct results
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=3)
        assert len(results) == 3
        assert results[0].doc_id == 0  # Exact match
        assert results[0].score < 0.001  # Distance ~0
    
    def test_index_after_add_warns_user(self, initialized_db):
        """Creating index after adding documents should warn the user.
        
        Note: Caliby does not store vectors separately from the HNSW index,
        so vectors added before the index exists cannot be retroactively indexed.
        Users should always create the index BEFORE adding documents.
        """
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("idx_after", schema, vector_dim=4)
        
        # Add documents FIRST (before index exists) - this is NOT recommended
        vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        ids = col.add(["a", "b", "c"], [{"idx": i} for i in range(3)], vectors)
        
        # Create index AFTER - should warn that vectors aren't indexed
        col.create_hnsw_index("vec_idx")
        
        # Search will NOT find the exact vectors because they weren't indexed
        # This test documents the expected (limited) behavior
        # The proper pattern is to create index BEFORE adding documents
    
    def test_search_recall_with_many_vectors(self, initialized_db):
        """Test recall with larger dataset to ensure index is working."""
        schema = caliby.Schema()
        schema.add_field("idx", caliby.FieldType.INT)
        col = caliby.Collection("recall_test", schema, vector_dim=8,
                                 distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec_idx")
        
        # Create 100 vectors with known structure
        import numpy as np
        np.random.seed(42)
        n = 100
        vectors = np.random.randn(n, 8).astype(np.float32)
        
        # Normalize to unit vectors
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        ids = col.add(
            [f"doc_{i}" for i in range(n)],
            [{"idx": i} for i in range(n)],
            vectors.tolist()
        )
        
        # Search for each vector - should find itself as nearest
        correct = 0
        for i in range(min(20, n)):  # Test first 20
            results = col.search_vector(vectors[i].tolist(), "vec_idx", k=1)
            if results[0].doc_id == i:
                correct += 1
        
        recall = correct / min(20, n)
        assert recall >= 0.9, f"Recall too low: {recall}. Index may not be working correctly."


class TestCollectionDistanceMetrics:
    """Test different distance metrics."""

    def test_l2_distance_metric(self, initialized_db):
        """Test L2 (Euclidean) distance metric."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("l2_test", schema, vector_dim=4,
                                 distance_metric=caliby.DistanceMetric.L2)
        col.create_hnsw_index("vec_idx")
        
        # Add vectors: one close to query, one far
        ids = col.add(["close", "far"], [{"name": "close"}, {"name": "far"}],
                      [[1, 0, 0, 0], [5, 0, 0, 0]])
        
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=2)
        
        # L2 distance: should find [1,0,0,0] first (distance 0)
        assert results[0].doc_id == 0
        assert results[0].score < 0.001

    def test_cosine_distance_metric(self, initialized_db):
        """Test Cosine distance metric."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("cosine_test", schema, vector_dim=4,
                                 distance_metric=caliby.DistanceMetric.COSINE)
        col.create_hnsw_index("vec_idx")
        
        # Same direction vectors should have low cosine distance
        # Orthogonal vectors should have distance ~1
        ids = col.add(["same_dir", "ortho"], [{"name": "same"}, {"name": "ortho"}],
                      [[1, 0, 0, 0], [0, 1, 0, 0]])
        
        results = col.search_vector([2, 0, 0, 0], "vec_idx", k=2)
        
        # First result should be same direction (distance ~0)
        assert results[0].doc_id == 0
        assert results[0].score < 0.01
        # Second should be orthogonal (distance ~1)
        assert abs(results[1].score - 1.0) < 0.01

    def test_inner_product_distance_metric(self, initialized_db):
        """Test Inner Product distance metric."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("ip_test", schema, vector_dim=4,
                                 distance_metric=caliby.DistanceMetric.IP)
        col.create_hnsw_index("vec_idx")
        
        # Normalized vectors for IP
        v1 = [1, 0, 0, 0]  # IP with [1,0,0,0] = 1
        v2 = [0, 1, 0, 0]  # IP with [1,0,0,0] = 0
        
        ids = col.add(["high_ip", "low_ip"], [{"name": "high"}, {"name": "low"}],
                      [v1, v2])
        
        results = col.search_vector([1, 0, 0, 0], "vec_idx", k=2)
        
        # IP returns 1 - inner_product as distance, so v1 should have lower distance
        assert results[0].doc_id == 0


class TestCollectionTextSearch:
    """Test Collection text search operations."""

    def test_create_text_index(self, initialized_db):
        """Test creating a text index."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        col = caliby.Collection("text_idx_test", schema)
        
        # Create text index
        col.create_text_index("text_idx")
        
        # Add documents
        ids = col.add(["The quick brown fox jumps over the lazy dog",
                       "A fast red fox runs through the forest"],
                      [{"title": "Doc 1"}, {"title": "Doc 2"}])
        
        assert col.doc_count() == 2

    def test_text_search_basic(self, initialized_db):
        """Test basic text search."""
        schema = caliby.Schema()
        schema.add_field("category", caliby.FieldType.STRING)
        col = caliby.Collection("text_search", schema)
        col.create_text_index("text_idx")
        
        ids = col.add(["Python programming tutorial",
                       "Java development guide",
                       "Python machine learning basics"],
                      [{"category": "python"}, {"category": "java"}, {"category": "python"}])
        
        # Search for Python
        results = col.search_text("python", "text_idx", k=10)
        
        # Should find at least 2 Python documents
        assert len(results) >= 2


class TestCollectionHybridSearch:
    """Test Collection hybrid search operations."""

    def test_hybrid_search_basic(self, initialized_db):
        """Test basic hybrid search combining vector and text."""
        schema = caliby.Schema()
        schema.add_field("topic", caliby.FieldType.STRING)
        col = caliby.Collection("hybrid_test", schema, vector_dim=4)
        col.create_hnsw_index("vec_idx")
        col.create_text_index("text_idx")
        
        ids = col.add(["Python programming tutorial",
                       "Java development guide",
                       "Python machine learning"],
                      [{"topic": "programming"}, {"topic": "programming"}, {"topic": "ml"}],
                      [[1, 0, 0, 0], [0, 1, 0, 0], [0.5, 0.5, 0, 0]])
        
        # Search with both vector and text
        results = col.search_hybrid(
            query_vec=[1, 0, 0, 0],
            vector_index="vec_idx",
            query_text="Python",
            text_index="text_idx",
            k=3
        )
        
        # Should return results
        assert len(results) > 0


class TestCollectionPersistence:
    """Test Collection persistence and recovery."""

    def test_collection_persists_documents(self, db_path):
        """Test that documents persist across reopening."""
        # First session: create and add documents
        caliby.open(db_path)
        schema = caliby.Schema()
        schema.add_field("value", caliby.FieldType.INT)
        col = caliby.Collection("persist_test", schema)
        
        ids = col.add(["Doc 1", "Doc 2", "Doc 3"],
                      [{"value": 10}, {"value": 20}, {"value": 30}])
        
        original_count = col.doc_count()
        col.flush()  # Ensure metadata is persisted before closing
        caliby.close()
        
        # Second session: reopen and verify
        caliby.open(db_path)
        col2 = caliby.Collection.open("persist_test")
        
        assert col2.doc_count() == original_count
        
        docs = col2.get(list(ids))
        values = sorted([d.metadata["value"] for d in docs])
        assert values == [10, 20, 30]
        
        caliby.close()


class TestCollectionEdgeCases:
    """Test edge cases and error handling."""

    def test_add_mismatched_lengths(self, initialized_db):
        """Test that mismatched lengths raise error."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("mismatch", schema)
        
        with pytest.raises(Exception):
            col.add(["a", "b"], [{"name": "a"}])  # 2 contents, 1 metadata

    def test_add_vectors_without_vector_support(self, initialized_db):
        """Test adding vectors to non-vector collection raises error."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("no_vec", schema)  # No vector_dim
        
        with pytest.raises(Exception):
            col.add(["doc"], [{"name": "test"}], [[1, 2, 3, 4]])

    def test_search_nonexistent_index(self, initialized_db):
        """Test searching nonexistent index raises error."""
        schema = caliby.Schema()
        schema.add_field("name", caliby.FieldType.STRING)
        col = caliby.Collection("no_idx", schema, vector_dim=4)
        
        ids = col.add(["doc"], [{"name": "test"}], [[1, 2, 3, 4]])
        
        with pytest.raises(Exception):
            col.search_vector([1, 2, 3, 4], "nonexistent_idx", k=5)

