"""
Tests for all code examples in docs/USAGE.md

This test file verifies that every code example in the usage documentation
works correctly. Each test corresponds to a section in the docs.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import json
import caliby


class TestQuickStart:
    """Test the Quick Start example from docs/USAGE.md"""

    def test_quick_start_example(self):
        """Test the complete quick start workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            # 1. Configure and initialize the system
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # 2. Create a collection with schema
                schema = caliby.Schema()
                schema.add_field("title", caliby.FieldType.STRING)
                schema.add_field("category", caliby.FieldType.STRING)

                collection = caliby.Collection("my_docs", schema, vector_dim=128)
                
                # Create indexes first (before adding data)
                collection.create_hnsw_index("vec_idx")
                collection.create_text_index("text_idx")

                # 3. Add documents with vectors
                contents = ["First document", "Second document", "Third document"]
                metadatas = [
                    {"title": "Doc 1", "category": "tech"},
                    {"title": "Doc 2", "category": "science"},
                    {"title": "Doc 3", "category": "tech"}
                ]
                vectors = np.random.rand(3, 128).astype(np.float32).tolist()

                ids = collection.add(contents, metadatas, vectors)
                
                assert collection.doc_count() == 3

                # 4. Search
                query_vector = np.random.rand(128).astype(np.float32)
                results = collection.search_vector(query_vector, "vec_idx", k=10)

                # Verify we got results
                assert isinstance(results, list)
                assert len(results) <= 10
                
                for r in results:
                    assert hasattr(r, 'doc_id')
                    assert hasattr(r, 'score')

            finally:
                # 6. Close when done
                caliby.close()


class TestSystemConfiguration:
    """Test System Configuration examples from docs/USAGE.md"""

    def test_buffer_pool_configuration(self):
        """Test buffer pool configuration."""
        # Set buffer pool size (in GB)
        caliby.set_buffer_config(size_gb=0.5)
        
        # Optional: set virtual memory limit
        caliby.set_buffer_config(size_gb=0.5, virtgb=2.0)
        # No assertion needed - just verify it doesn't crash

    def test_initialize_open_data_directory(self):
        """Test opening data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data")
            
            caliby.set_buffer_config(size_gb=0.5)
            
            # Open data directory (creates if doesn't exist)
            caliby.open(data_path)
            caliby.close()
            
            # Force cleanup existing data
            caliby.open(data_path, cleanup_if_exist=True)
            caliby.close()

    def test_shutdown_and_flush(self):
        """Test shutdown and flush operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_path)
            
            # Just flush without closing
            caliby.flush_storage()
            
            # Flush and close all resources
            caliby.close()


class TestHNSWIndex:
    """Test HNSW Index examples from docs/USAGE.md"""

    def test_hnsw_index_full_example(self):
        """Test the complete HNSW index workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            # Initialize system
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # Create HNSW index
                index = caliby.HnswIndex(
                    max_elements=100_000,     # Reduced for testing
                    dim=128,
                    M=16,
                    ef_construction=200,
                    enable_prefetch=True,
                    skip_recovery=True,       # Skip recovery for clean test
                    index_id=0,
                    name='my_index'
                )

                # Add vectors (batch operation) - reduced size for testing
                vectors = np.random.rand(1000, 128).astype(np.float32)
                index.add_points(vectors, num_threads=4)

                # Single query search
                query = np.random.rand(128).astype(np.float32)
                labels, distances = index.search_knn(query, k=10, ef_search=100)

                assert len(labels) == 10
                assert len(distances) == 10

                # Batch search (parallel)
                queries = np.random.rand(10, 128).astype(np.float32)
                labels, distances = index.search_knn_parallel(
                    queries, k=10, ef_search=100, num_threads=4
                )

                assert labels.shape == (10, 10)
                assert distances.shape == (10, 10)

                # Get index info
                assert index.get_name() == 'my_index'
                assert index.get_dim() == 128
                assert isinstance(index.was_recovered(), bool)

                # Get statistics
                stats = index.get_stats()
                assert 'dist_comps' in stats
                assert 'num_levels' in stats

                # Flush to storage
                index.flush()

            finally:
                caliby.close()


class TestDiskANNIndex:
    """Test DiskANN Index examples from docs/USAGE.md"""

    def test_diskann_index_full_example(self):
        """Test the complete DiskANN index workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # Create DiskANN index
                index = caliby.DiskANN(
                    dimensions=128,
                    max_elements=100_000,
                    R_max_degree=64,
                    is_dynamic=True
                )

                # Prepare data with tags (for filtering) - reduced size for testing
                num_vectors = 1000
                vectors = np.random.rand(num_vectors, 128).astype(np.float32)
                tags = [[i % 100] for i in range(num_vectors)]

                # Build parameters
                params = caliby.BuildParams()
                params.L_build = 100
                params.alpha = 1.2
                params.num_threads = 4

                # Build the index
                index.build(vectors, tags, params)

                # Search parameters
                search_params = caliby.SearchParams(L_search=50)
                search_params.beam_width = 4

                # Basic search
                query = np.random.rand(128).astype(np.float32)
                labels, distances = index.search(query, K=10, params=search_params)

                assert len(labels) <= 10
                assert len(distances) <= 10

                # Filtered search (only return vectors with tag=42)
                labels, distances = index.search_with_filter(
                    query, filter_label=42, K=10, params=search_params
                )

                # Parallel batch search
                queries = np.random.rand(10, 128).astype(np.float32)
                labels, distances = index.search_knn_parallel(
                    queries, K=10, params=search_params, num_threads=4
                )

                assert labels.shape[0] == 10
                assert distances.shape[0] == 10

                # Dynamic operations (if is_dynamic=True)
                new_point = np.random.rand(128).astype(np.float32)
                index.insert_point(new_point, tags=[99], external_id=num_vectors)
                index.lazy_delete(external_id=num_vectors)
                # Note: consolidate_deletes is not fully implemented yet in the optimized version
                # index.consolidate_deletes(params)

            finally:
                caliby.close()


class TestIVFPQIndex:
    """Test IVF+PQ Index examples from docs/USAGE.md"""

    def test_ivfpq_index_full_example(self):
        """Test the complete IVF+PQ index workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # Create IVF+PQ index - reduced sizes for testing
                index = caliby.IVFPQIndex(
                    max_elements=100_000,
                    dim=128,
                    num_clusters=64,           # Reduced for testing
                    num_subquantizers=8,
                    retrain_interval=10000,
                    skip_recovery=True,
                    index_id=0,
                    name='large_dataset'
                )

                # IMPORTANT: Train the index first
                training_data = np.random.rand(5000, 128).astype(np.float32)
                index.train(training_data)

                assert index.is_trained() == True

                # Add points (after training) - reduced size for testing
                vectors = np.random.rand(10000, 128).astype(np.float32)
                index.add_points(vectors, num_threads=4)

                # Search with nprobe parameter
                query = np.random.rand(128).astype(np.float32)
                labels, distances = index.search_knn(query, k=10, nprobe=8)

                assert len(labels) <= 10
                assert len(distances) <= 10

                # Parallel batch search
                queries = np.random.rand(10, 128).astype(np.float32)
                labels, distances = index.search_knn_parallel(
                    queries, k=10, nprobe=16, num_threads=4
                )

                assert labels.shape == (10, 10)
                assert distances.shape == (10, 10)

                # Get statistics
                stats = index.get_stats()
                assert 'num_clusters' in stats
                assert 'avg_list_size' in stats

            finally:
                caliby.close()


class TestSchemaDefinition:
    """Test Schema Definition examples from docs/USAGE.md"""

    def test_schema_creation_and_fields(self):
        """Test schema creation with all field types."""
        schema = caliby.Schema()

        # Add fields with different types
        schema.add_field("title", caliby.FieldType.STRING)
        schema.add_field("year", caliby.FieldType.INT)
        schema.add_field("rating", caliby.FieldType.FLOAT)
        schema.add_field("published", caliby.FieldType.BOOL)
        schema.add_field("tags", caliby.FieldType.STRING_ARRAY)
        schema.add_field("scores", caliby.FieldType.INT_ARRAY)

        # Optional: nullable parameter (default True)
        schema.add_field("optional_field", caliby.FieldType.STRING, nullable=True)

        # Check fields
        assert schema.has_field("title") == True
        assert schema.has_field("nonexistent") == False
        
        fields = schema.fields()
        assert len(fields) == 7


class TestDocumentOperations:
    """Test Document Operations examples from docs/USAGE.md"""

    def test_document_crud_operations(self):
        """Test document CRUD operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                schema = caliby.Schema()
                schema.add_field("title", caliby.FieldType.STRING)
                schema.add_field("category", caliby.FieldType.STRING)
                schema.add_field("year", caliby.FieldType.INT)

                # Create collection (without vectors)
                collection = caliby.Collection("docs", schema)

                # Or with vector support
                collection_with_vec = caliby.Collection(
                    "vec_docs", schema, 
                    vector_dim=128,
                    distance_metric=caliby.DistanceMetric.COSINE
                )

                # Add documents (batch)
                contents = ["Doc one", "Doc two", "Doc three", "Doc four", "Doc five"]
                metadatas = [
                    {"title": "First", "category": "A", "year": 2020},
                    {"title": "Second", "category": "B", "year": 2021},
                    {"title": "Third", "category": "A", "year": 2022},
                    {"title": "Fourth", "category": "B", "year": 2023},
                    {"title": "Fifth", "category": "A", "year": 2024}
                ]

                ids = collection.add(contents, metadatas)

                assert collection.doc_count() == 5

                # Get documents by ID
                docs = collection.get([ids[0], ids[2], ids[4]])
                assert len(docs) == 3
                
                for doc in docs:
                    assert hasattr(doc, 'id')
                    assert hasattr(doc, 'content')
                    assert hasattr(doc, 'metadata')

                # Update metadata
                collection.update([ids[0], ids[1]], [
                    {"title": "Updated First", "category": "A", "year": 2025},
                    {"title": "Updated Second", "category": "C", "year": 2025}
                ])

                # Verify update
                updated_docs = collection.get([ids[0], ids[1]])
                doc1 = [d for d in updated_docs if d.id == ids[0]][0]
                assert doc1.metadata["title"] == "Updated First"

                # Delete documents
                collection.delete([ids[3], ids[4]])
                assert collection.doc_count() == 3

                # Open existing collection
                existing = caliby.Collection.open("docs")
                assert existing.doc_count() == 3

            finally:
                caliby.close()


class TestVectorOperations:
    """Test Vector Operations examples from docs/USAGE.md"""

    def test_vector_operations(self):
        """Test vector operations on collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                schema = caliby.Schema()
                schema.add_field("name", caliby.FieldType.STRING)

                # Create collection with vectors
                collection = caliby.Collection("vectors", schema, vector_dim=64)

                # Create HNSW index first (before adding data)
                collection.create_hnsw_index("vec_idx", M=16, ef_construction=200)

                # Add documents with vectors
                contents = ["First", "Second", "Third"]
                metadatas = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
                vectors = np.random.rand(3, 64).astype(np.float32).tolist()

                ids = collection.add(contents, metadatas, vectors)

                # Add more documents with vectors
                more_contents = ["Fourth", "Fifth"]
                more_metadatas = [{"name": "D"}, {"name": "E"}]
                more_vectors = np.random.rand(2, 64).astype(np.float32).tolist()
                more_ids = collection.add(more_contents, more_metadatas, more_vectors)

                assert collection.doc_count() == 5

                # Or DiskANN index
                collection.create_diskann_index("disk_idx", R=64, L=100, alpha=1.2)

                # Vector search
                query = np.random.rand(64).astype(np.float32)
                results = collection.search_vector(query, "vec_idx", k=5)

                assert len(results) <= 5
                for r in results:
                    assert hasattr(r, 'doc_id')
                    assert hasattr(r, 'score')

            finally:
                caliby.close()


class TestTextSearch:
    """Test Text Search (BM25) examples from docs/USAGE.md"""

    def test_text_search_bm25(self):
        """Test BM25 text search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                schema = caliby.Schema()
                schema.add_field("title", caliby.FieldType.STRING)

                collection = caliby.Collection("articles", schema)

                # Create text index first (before adding documents)
                collection.create_text_index("text_idx")

                # Add documents
                contents = [
                    "Machine learning is a subset of artificial intelligence",
                    "Deep learning uses neural networks for pattern recognition",
                    "Natural language processing enables text understanding",
                    "Computer vision helps machines interpret images"
                ]
                metadatas = [
                    {"title": "ML Intro"},
                    {"title": "Deep Learning"},
                    {"title": "NLP Guide"},
                    {"title": "CV Tutorial"}
                ]

                ids = collection.add(contents, metadatas)

                # Text search
                results = collection.search_text("machine learning neural", "text_idx", k=3)

                assert len(results) <= 3
                for r in results:
                    assert hasattr(r, 'doc_id')
                    assert hasattr(r, 'text_score')

            finally:
                caliby.close()


class TestHybridSearch:
    """Test Hybrid Search examples from docs/USAGE.md"""

    def test_hybrid_search(self):
        """Test hybrid vector + text search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                schema = caliby.Schema()
                schema.add_field("title", caliby.FieldType.STRING)

                collection = caliby.Collection("hybrid_docs", schema, vector_dim=128)

                # Create indexes first (before adding data)
                collection.create_hnsw_index("vec_idx")
                collection.create_text_index("text_idx")

                # Add documents with vectors
                contents = [
                    "Introduction to machine learning algorithms",
                    "Deep neural network architectures",
                    "Statistical methods in data science"
                ]
                metadatas = [{"title": f"Doc {i}"} for i in range(3)]
                vectors = np.random.rand(3, 128).astype(np.float32).tolist()

                ids = collection.add(contents, metadatas, vectors)

                # Configure fusion parameters - RRF
                fusion = caliby.FusionParams()
                fusion.method = caliby.FusionMethod.RRF
                fusion.rrf_k = 60

                # Hybrid search
                query_vec = np.random.rand(128).astype(np.float32)
                query_text = "machine learning"

                results = collection.search_hybrid(
                    query_vec, "vec_idx",
                    query_text, "text_idx",
                    k=5,
                    fusion=fusion
                )

                assert len(results) <= 5
                for r in results:
                    assert hasattr(r, 'doc_id')
                    assert hasattr(r, 'score')
                    assert hasattr(r, 'vector_score')
                    assert hasattr(r, 'text_score')

                # Test weighted fusion
                fusion_weighted = caliby.FusionParams()
                fusion_weighted.method = caliby.FusionMethod.WEIGHTED
                fusion_weighted.vector_weight = 0.7
                fusion_weighted.text_weight = 0.3

                results_weighted = collection.search_hybrid(
                    query_vec, "vec_idx",
                    query_text, "text_idx",
                    k=5,
                    fusion=fusion_weighted
                )

                assert len(results_weighted) <= 5

            finally:
                caliby.close()


class TestMetadataIndexing:
    """Test Metadata Indexing examples from docs/USAGE.md"""

    def test_metadata_indexing(self):
        """Test metadata index creation and operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                schema = caliby.Schema()
                schema.add_field("category", caliby.FieldType.STRING)
                schema.add_field("year", caliby.FieldType.INT)
                schema.add_field("price", caliby.FieldType.FLOAT)

                collection = caliby.Collection("products", schema)

                # Single-field index
                collection.create_metadata_index("year_idx", ["year"])

                # Composite index
                collection.create_metadata_index("category_year_idx", ["category", "year"])

                # Unique index
                collection.create_metadata_index(
                    "unique_idx", ["category", "year"], unique=True
                )

                # Legacy API (single field only)
                collection.create_btree_index("price_btree", "price")

                # List all indexes
                indexes = collection.list_indices()
                assert len(indexes) >= 4
                
                for idx in indexes:
                    assert 'name' in idx
                    assert 'type' in idx
                    assert 'config' in idx

                # Drop an index
                collection.drop_index("price_btree")
                
                # Verify dropped
                indexes_after = collection.list_indices()
                index_names = [idx['name'] for idx in indexes_after]
                assert "price_btree" not in index_names

            finally:
                caliby.close()


class TestFiltering:
    """Test Filtering examples from docs/USAGE.md"""

    def test_filtering_operations(self):
        """Test filter operations on search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                schema = caliby.Schema()
                schema.add_field("category", caliby.FieldType.STRING)
                schema.add_field("year", caliby.FieldType.INT)
                schema.add_field("price", caliby.FieldType.FLOAT)
                schema.add_field("active", caliby.FieldType.BOOL)

                collection = caliby.Collection("filtered", schema, vector_dim=64)

                # Create indices first (before adding data)
                collection.create_hnsw_index("vec_idx")
                collection.create_text_index("text_idx")

                # Add sample data
                n = 100
                contents = [f"Product {i}" for i in range(n)]
                metadatas = [
                    {
                        "category": ["tech", "home", "office"][i % 3],
                        "year": 2020 + (i % 5),
                        "price": 10.0 + (i * 0.5),
                        "active": i % 2 == 0
                    }
                    for i in range(n)
                ]
                vectors = np.random.rand(n, 64).astype(np.float32).tolist()

                ids = collection.add(contents, metadatas, vectors)

                # Test all filter types

                # Simple equality filter
                filter1 = json.dumps({"field": "category", "op": "eq", "value": "tech"})

                # Comparison filter
                filter2 = json.dumps({"field": "year", "op": "gte", "value": 2023})

                # AND condition
                filter3 = json.dumps({
                    "and": [
                        {"field": "category", "op": "eq", "value": "tech"},
                        {"field": "price", "op": "lt", "value": 50.0}
                    ]
                })

                # OR condition
                filter4 = json.dumps({
                    "or": [
                        {"field": "category", "op": "eq", "value": "tech"},
                        {"field": "category", "op": "eq", "value": "home"}
                    ]
                })

                # NOT condition
                filter5 = json.dumps({
                    "not": {"field": "active", "op": "eq", "value": False}
                })

                # IN operator
                filter6 = json.dumps({
                    "field": "year", "op": "in", "value": [2022, 2023, 2024]
                })

                # Apply filter to vector search
                query = np.random.rand(64).astype(np.float32)
                results = collection.search_vector(query, "vec_idx", k=10, filter=filter3)
                assert isinstance(results, list)

                # Apply filter to text search
                results = collection.search_text("product", "text_idx", k=10, filter=filter1)
                assert isinstance(results, list)

                # Apply filter to hybrid search
                fusion = caliby.FusionParams()
                results = collection.search_hybrid(
                    query, "vec_idx",
                    "product", "text_idx",
                    k=10, fusion=fusion, filter=filter4
                )
                assert isinstance(results, list)

            finally:
                caliby.close()


class TestIndexCatalog:
    """Test Index Catalog examples from docs/USAGE.md"""

    def test_index_catalog_operations(self):
        """Test index catalog operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # Get the singleton catalog instance
                catalog = caliby.IndexCatalog.instance()

                assert catalog.is_initialized() == True
                assert catalog.data_dir() == data_dir

                # Create indexes with simplified API
                hnsw_handle = catalog.create_hnsw_index(
                    name="embeddings",
                    dimensions=128,
                    max_elements=100000,
                    M=16,
                    ef_construction=200
                )

                diskann_handle = catalog.create_diskann_index(
                    name="vectors",
                    dimensions=256,
                    max_elements=50000,
                    R_max_degree=64,
                    L_build=100,
                    alpha=1.2
                )

                # List all indexes
                indexes = catalog.list_indexes()
                assert len(indexes) >= 2
                
                for info in indexes:
                    assert hasattr(info, 'name')
                    assert hasattr(info, 'type')
                    assert hasattr(info, 'num_elements')

                # Check if index exists
                assert catalog.index_exists("embeddings") == True
                assert catalog.index_exists("nonexistent") == False

                # Get detailed info
                info = catalog.get_index_info("embeddings")
                assert hasattr(info, 'create_time')
                assert hasattr(info, 'file_path')

                # Open existing index
                handle = catalog.open_index("embeddings")
                assert handle.is_valid()

                # Flush changes
                handle.flush()

                # Drop index
                catalog.drop_index("vectors")
                assert catalog.index_exists("vectors") == False

            finally:
                caliby.close()


class TestPersistenceAndRecovery:
    """Test Persistence & Recovery examples from docs/USAGE.md"""

    def test_persistence_and_recovery(self):
        """Test data persistence and recovery across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "persistent_data")
            
            # First session: create and populate
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            schema = caliby.Schema()
            schema.add_field("name", caliby.FieldType.STRING)

            collection = caliby.Collection("persistent", schema, vector_dim=64)
            collection.create_hnsw_index("vec_idx")
            ids = collection.add(["A", "B", "C"], 
                           [{"name": n} for n in ["a", "b", "c"]],
                           np.random.rand(3, 64).astype(np.float32).tolist())

            # Explicit flush
            collection.flush()

            caliby.close()

            # Second session: recovery
            # Note: buffer config persists across close/open in same process
            caliby.open(data_dir)

            # Open existing collection
            recovered = caliby.Collection.open("persistent")
            assert recovered.doc_count() == 3

            # HNSW index also supports recovery
            index = caliby.HnswIndex(
                max_elements=1_000_000,
                dim=64,
                M=16,
                ef_construction=200,
                enable_prefetch=True,
                skip_recovery=False,  # Enable recovery
                index_id=1,
                name='my_index'
            )

            # was_recovered() returns whether existing data was found
            assert isinstance(index.was_recovered(), bool)

            caliby.close()


class TestPerformanceTips:
    """Test Performance Tips examples from docs/USAGE.md"""

    def test_buffer_pool_sizing(self):
        """Test buffer pool sizing configuration."""
        # For a 10GB dataset
        caliby.set_buffer_config(size_gb=4.0)  # 40% of dataset
        # No assertion needed - just verify it doesn't crash

    def test_hnsw_tuning_configurations(self):
        """Test HNSW tuning configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # High recall configuration
                index_high_recall = caliby.HnswIndex(
                    max_elements=10000,
                    dim=128,
                    M=32,
                    ef_construction=400,
                    enable_prefetch=True,
                    skip_recovery=True,
                    index_id=0,
                    name='high_recall'
                )

                vectors = np.random.rand(100, 128).astype(np.float32)
                index_high_recall.add_points(vectors, num_threads=4)

                query = np.random.rand(128).astype(np.float32)
                results = index_high_recall.search_knn(query, k=10, ef_search=200)
                assert len(results[0]) == 10

                # Fast search configuration
                index_fast = caliby.HnswIndex(
                    max_elements=10000,
                    dim=128,
                    M=16,
                    ef_construction=200,
                    enable_prefetch=True,
                    skip_recovery=True,
                    index_id=1,
                    name='fast_search'
                )

                index_fast.add_points(vectors, num_threads=4)
                results = index_fast.search_knn(query, k=10, ef_search=50)
                assert len(results[0]) == 10

            finally:
                caliby.close()

    def test_ivfpq_tuning_configuration(self):
        """Test IVF+PQ tuning configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # For 1M vectors (but we use smaller size for testing)
                index = caliby.IVFPQIndex(
                    max_elements=100_000,  # Reduced for testing
                    dim=128,
                    num_clusters=100,      # Reduced for testing
                    num_subquantizers=16
                )

                # Train and add
                training_data = np.random.rand(1000, 128).astype(np.float32)
                index.train(training_data)
                
                vectors = np.random.rand(1000, 128).astype(np.float32)
                index.add_points(vectors, num_threads=4)

                query = np.random.rand(128).astype(np.float32)
                labels, distances = index.search_knn(query, k=10, nprobe=8)
                assert len(labels) <= 10

            finally:
                caliby.close()

    def test_parallel_operations(self):
        """Test parallel insertion and search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                index = caliby.HnswIndex(
                    max_elements=100000,
                    dim=128,
                    M=16,
                    ef_construction=200,
                    enable_prefetch=True,
                    skip_recovery=True,
                    index_id=0,
                    name='parallel_test'
                )

                vectors = np.random.rand(1000, 128).astype(np.float32)
                
                # Parallel insertion
                index.add_points(vectors, num_threads=8)

                # Parallel search
                queries = np.random.rand(10, 128).astype(np.float32)
                results = index.search_knn_parallel(queries, k=10, ef_search=100, num_threads=8)

                assert results[0].shape == (10, 10)
                assert results[1].shape == (10, 10)

            finally:
                caliby.close()

    def test_force_eviction(self):
        """Test force eviction for memory-constrained scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "caliby_data")
            
            caliby.set_buffer_config(size_gb=0.5)
            caliby.open(data_dir)

            try:
                # Create some data to evict
                index = caliby.HnswIndex(
                    max_elements=10000,
                    dim=128,
                    M=16,
                    ef_construction=200,
                    enable_prefetch=True,
                    skip_recovery=True,
                    index_id=0,
                    name='eviction_test'
                )

                vectors = np.random.rand(1000, 128).astype(np.float32)
                index.add_points(vectors, num_threads=4)

                # Evict a portion of buffer pool (0.0 to 1.0)
                caliby.force_evict_buffer_portion(0.5)  # Evict 50%

                # Verify index still works after eviction
                query = np.random.rand(128).astype(np.float32)
                labels, distances = index.search_knn(query, k=10, ef_search=100)
                assert len(labels) == 10

            finally:
                caliby.close()


class TestDistanceMetrics:
    """Test distance metrics enumeration."""

    def test_distance_metrics_enum(self):
        """Test that all distance metrics are accessible."""
        assert caliby.DistanceMetric.L2 is not None
        assert caliby.DistanceMetric.COSINE is not None
        assert caliby.DistanceMetric.IP is not None


class TestFieldTypes:
    """Test field types enumeration."""

    def test_field_types_enum(self):
        """Test that all field types are accessible."""
        assert caliby.FieldType.STRING is not None
        assert caliby.FieldType.INT is not None
        assert caliby.FieldType.FLOAT is not None
        assert caliby.FieldType.BOOL is not None
        assert caliby.FieldType.STRING_ARRAY is not None
        assert caliby.FieldType.INT_ARRAY is not None


class TestFusionMethods:
    """Test fusion methods enumeration."""

    def test_fusion_methods_enum(self):
        """Test that all fusion methods are accessible."""
        assert caliby.FusionMethod.RRF is not None
        assert caliby.FusionMethod.WEIGHTED is not None


class TestIndexTypes:
    """Test index types enumeration."""

    def test_index_types_enum(self):
        """Test that all index types are accessible."""
        assert caliby.IndexType.HNSW is not None
        assert caliby.IndexType.DISKANN is not None
        assert caliby.IndexType.IVF is not None
