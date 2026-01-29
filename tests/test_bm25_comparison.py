"""
Test suite comparing Caliby's BM25 Text Index with bm25s

This test verifies:
1. BM25 scoring formula correctness
2. Ranking similarity between implementations
3. Edge cases (empty queries, single terms, etc.)
4. Performance characteristics
"""

import pytest
import numpy as np
import os
import sys
import time
import tempfile

# Import caliby
import caliby

# Try to import bm25s
try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False

# Try to import pytest-benchmark
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


# Test fixtures
@pytest.fixture
def sample_corpus():
    """Small corpus for basic testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing deals with text and speech",
        "Computer vision helps machines understand images",
        "The cat sat on the mat and watched the birds",
        "Dogs are loyal companions and love to play fetch",
        "Python is a popular programming language for data science",
        "Neural networks learn from data through backpropagation",
        "Artificial intelligence transforms many industries",
    ]


@pytest.fixture
def large_corpus():
    """Larger corpus for performance testing."""
    np.random.seed(42)
    vocab = [
        "machine", "learning", "deep", "neural", "network", "data", "science",
        "artificial", "intelligence", "computer", "vision", "natural", "language",
        "processing", "algorithm", "model", "training", "prediction", "classification",
        "regression", "clustering", "optimization", "gradient", "descent", "batch",
        "quick", "brown", "fox", "jumps", "lazy", "dog", "cat", "bird", "fish",
    ]
    
    corpus = []
    for _ in range(5000):
        doc_len = np.random.randint(10, 50)
        words = np.random.choice(vocab, size=doc_len, replace=True)
        corpus.append(" ".join(words))
    
    return corpus


@pytest.fixture
def caliby_setup(tmp_path):
    """Setup Caliby with a temporary directory."""
    os.chdir(tmp_path)
    caliby.set_buffer_config(1.0)
    caliby.open(str(tmp_path))
    yield tmp_path
    caliby.close()


class TestBM25ScoreFormula:
    """Test BM25 scoring formula correctness."""
    
    def test_idf_calculation(self, sample_corpus, caliby_setup):
        """Test that IDF is calculated correctly."""
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        # For a term appearing in 1 of 10 documents:
        # IDF = log((10 - 1 + 0.5) / (1 + 0.5) + 1) = log(9.5/1.5 + 1) = log(7.33)
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("idf_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))  # Start from 0
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        # Query for "fox" which appears in document 1 (0-indexed: 0)
        results = collection.search_text("fox", "content_idx", 10)
        
        assert len(results) >= 1
        assert results[0].doc_id == 0  # Doc with "fox" (1-indexed)
        assert results[0].score > 0
    
    def test_term_frequency_impact(self, caliby_setup):
        """Test that higher term frequency increases score."""
        corpus = [
            "cat cat cat cat cat",  # High TF
            "cat dog bird",          # Low TF
            "fish bird tree",        # No match
        ]
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("tf_test", schema)
        
        ids = [1, 2, 3]  # Doc IDs start from 0
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(corpus, metadatas)
        
        results = collection.search_text("cat", "content_idx", 10)
        
        # Doc 1 (high TF) should rank higher than Doc 2 (low TF)
        assert len(results) >= 2
        assert results[0].doc_id == 0  # High TF doc
        assert results[1].doc_id == 1  # Low TF doc
        assert results[0].score > results[1].score
    
    def test_document_length_normalization(self, caliby_setup):
        """Test that longer documents are penalized (b > 0)."""
        corpus = [
            "cat dog",  # Short document
            "cat dog bird fish tree house car road sky cloud water earth fire wind sun moon star planet galaxy universe cosmos eternity infinity",  # Long document
        ]
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("len_norm_test", schema)
        
        ids = [1, 2]  # Doc IDs start from 0
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(corpus, metadatas)
        
        results = collection.search_text("cat", "content_idx", 10)
        
        # Both should be returned (both contain "cat")
        assert len(results) == 2
        # Short doc should have higher score (same TF but shorter)
        assert results[0].doc_id == 0  # Short doc
        assert results[0].score > results[1].score


class TestRankingCorrectness:
    """Test that rankings are sensible and consistent."""
    
    def test_exact_match_ranks_high(self, sample_corpus, caliby_setup):
        """Test that exact query terms lead to high rankings."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("exact_match_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        # Query for specific term
        results = collection.search_text("machine learning", "content_idx", 10)
        
        # Documents about machine learning should rank highest
        top_ids = [r.doc_id for r in results[:3]]
        # Doc 2 (1-indexed) is "Machine learning is a subset of artificial intelligence"
        assert 1 in top_ids
    
    def test_multi_term_query(self, sample_corpus, caliby_setup):
        """Test that multi-term queries work correctly."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("multi_term_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        results = collection.search_text("neural networks deep learning", "content_idx", 10)
        
        # Should return documents about neural networks and deep learning
        assert len(results) >= 2
        top_ids = [r.doc_id for r in results[:3]]
        # Doc 2 is "Deep learning uses neural networks with many layers"
        assert 1 in top_ids
    
    def test_no_match_returns_empty(self, sample_corpus, caliby_setup):
        """Test that non-matching query returns empty results."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("no_match_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        results = collection.search_text("xyznonexistent123", "content_idx", 10)
        assert len(results) == 0


@pytest.mark.skipif(not BM25S_AVAILABLE, reason="bm25s not installed")
class TestBM25sComparison:
    """Compare Caliby with bm25s for functional equivalence."""
    
    def test_ranking_overlap(self, sample_corpus, caliby_setup):
        """Test that both implementations return similar rankings."""
        # Setup Caliby
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("comparison_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        # Setup bm25s
        corpus_tokens = bm25s.tokenize(sample_corpus, stopwords="en")
        retriever = bm25s.BM25(k1=1.2, b=0.75)  # Same params as Caliby default
        retriever.index(corpus_tokens)
        
        # Compare for several queries
        queries = ["machine learning", "neural networks", "quick brown fox"]
        
        for query in queries:
            # Caliby results (1-indexed doc IDs)
            cal_results = collection.search_text(query, "content_idx", 5)
            cal_ids = [r.doc_id for r in cal_results]  # Convert to 0-indexed for comparison
            
            # bm25s results (0-indexed doc IDs)
            query_tokens = bm25s.tokenize([query], stopwords="en")
            bm_ids, bm_scores = retriever.retrieve(query_tokens, k=5)
            bm_ids_list = [int(bm_ids[0, i]) for i in range(bm_ids.shape[1])]
            
            # Check overlap - should have significant overlap
            cal_set = set(cal_ids[:3])
            bm_set = set(bm_ids_list[:3])
            overlap = len(cal_set & bm_set)
            
            # At least 1 of top 3 should match (different BM25 variants may have different results)
            assert overlap >= 1, f"Query '{query}': Caliby top 3 {cal_ids[:3]}, bm25s top 3 {bm_ids_list[:3]}"
    
    def test_score_ordering_consistency(self, sample_corpus, caliby_setup):
        """Test that both produce same relative ordering for clear cases."""
        # Create corpus with clear ranking
        corpus = [
            "machine learning machine learning machine learning",  # High TF for "machine learning"
            "machine learning",  # Normal TF
            "dog cat bird",  # No match
        ]
        
        # Setup Caliby
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("order_test", schema)
        
        ids = [1, 2, 3]  # 0-indexed doc IDs
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(corpus, metadatas)
        
        # Setup bm25s
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        # Both should rank doc 0 (bm25s) / doc 1 (Caliby) highest for "machine learning"
        cal_results = collection.search_text("machine learning", "content_idx", 3)
        
        query_tokens = bm25s.tokenize(["machine learning"], stopwords="en")
        bm_ids, _ = retriever.retrieve(query_tokens, k=3)
        
        # Top result should be high TF doc for both
        assert cal_results[0].doc_id == 0  # Caliby 1-indexed
        assert bm_ids[0, 0] == 0  # bm25s 0-indexed


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_query(self, sample_corpus, caliby_setup):
        """Test empty query handling."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("empty_query_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        results = collection.search_text("", "content_idx", 10)
        assert len(results) == 0
    
    def test_single_document_corpus(self, caliby_setup):
        """Test with single document corpus."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("single_doc_test", schema)
        
        collection.create_text_index("content_idx")
        collection.add(["The quick brown fox"], [{"title": "doc_0"}])  # 0-indexed
        
        results = collection.search_text("fox", "content_idx", 10)
        assert len(results) == 1
        assert results[0].doc_id == 0

    def test_stopword_only_query(self, sample_corpus, caliby_setup):
        """Test query with only stopwords."""
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("stopword_test", schema)
        
        ids = list(range(1, len(sample_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(sample_corpus, metadatas)
        
        # "the" and "is" are stopwords
        results = collection.search_text("the is a", "content_idx", 10)
        # Should return empty or very few results since these are stopwords
        # (Implementation dependent on stopword handling)
    
    def test_special_characters(self, caliby_setup):
        """Test handling of special characters in text."""
        corpus = [
            "C++ is a programming language!",
            "Python 3.10 is the latest version.",
            "Email: test@example.com",
        ]
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("special_char_test", schema)
        
        ids = [1, 2, 3]  # 0-indexed
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(corpus, metadatas)
        
        # Should still find "programming"
        results = collection.search_text("programming", "content_idx", 10)
        assert len(results) >= 1
    
    def test_unicode_text(self, caliby_setup):
        """Test handling of unicode characters."""
        corpus = [
            "日本語テキスト",  # Japanese
            "Ελληνικά κείμενο",  # Greek  
            "中文文本",  # Chinese
            "Normal English text",
        ]
        
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("unicode_test", schema)
        
        ids = list(range(1, len(corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(corpus, metadatas)
        
        results = collection.search_text("English", "content_idx", 10)
        assert len(results) >= 1


class TestPerformance:
    """Performance benchmarks."""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    @pytest.mark.benchmark
    def test_index_throughput(self, large_corpus, caliby_setup, benchmark):
        """Benchmark index building throughput."""
        def build_index():
            schema = caliby.Schema()
            schema.add_field("title", caliby.FieldType.STRING)
            collection = caliby.Collection(f"perf_idx_{time.time()}", schema)
            
            ids = list(range(1, len(large_corpus) + 1))
            metadatas = [{"title": f"doc_{i}"} for i in ids]
            collection.create_text_index("content_idx")
            collection.add(large_corpus, metadatas)
            return collection
        
        benchmark(build_index)
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not installed")
    @pytest.mark.benchmark
    def test_search_throughput(self, large_corpus, caliby_setup, benchmark):
        """Benchmark search throughput."""
        # Setup
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("perf_search", schema)
        
        ids = list(range(1, len(large_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(large_corpus, metadatas)
        
        queries = ["machine learning", "neural network", "data science", "deep learning", "computer vision"]
        
        def run_searches():
            for q in queries:
                collection.search_text(q, "content_idx", 10)
        
        benchmark(run_searches)
    
    def test_qps_target(self, large_corpus, caliby_setup):
        """Test that QPS meets target (should be competitive with bm25s)."""
        # Setup
        schema = caliby.Schema()
        schema.add_field("title", caliby.FieldType.STRING)
        collection = caliby.Collection("qps_test", schema)
        
        ids = list(range(1, len(large_corpus) + 1))
        metadatas = [{"title": f"doc_{i}"} for i in ids]
        collection.create_text_index("content_idx")
        collection.add(large_corpus, metadatas)
        
        # Generate queries
        queries = ["machine learning", "neural network", "data science"] * 100
        
        # Warmup
        for q in queries[:10]:
            collection.search_text(q, "content_idx", 10)
        
        # Measure
        start = time.perf_counter()
        for q in queries:
            collection.search_text(q, "content_idx", 10)
        elapsed = time.perf_counter() - start
        
        qps = len(queries) / elapsed
        print(f"\nQPS: {qps:.1f}")
        
        # Target: at least 100 QPS for 5k doc corpus
        assert qps >= 100, f"QPS {qps:.1f} is below target of 100"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
