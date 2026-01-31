#!/usr/bin/env python3
"""
Semantic Search Example

This example demonstrates how to use Caliby for semantic search with:
- Full-text and vector search combined
- Faceted filtering
- Search result highlighting

Use Cases:
- Document search engines
- Knowledge base search
- Content discovery
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import caliby
import numpy as np
import shutil
import json
from typing import List, Dict, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer('all-MiniLM-L6-v2')
    USE_REAL_EMBEDDINGS = True
except ImportError:
    _model = None
    USE_REAL_EMBEDDINGS = False
    print("Warning: sentence-transformers not installed. Using simulated embeddings.")
    print("Install with: pip install sentence-transformers")
except Exception as e:
    _model = None
    USE_REAL_EMBEDDINGS = False
    print(f"Warning: Failed to load embedding model ({type(e).__name__}). Using simulated embeddings.")
    print("This may be due to network issues. The model will be cached after first successful download.")

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def search_result_to_dict(result) -> Dict:
    """Convert a caliby.SearchResult to a dictionary."""
    doc = result.document if result.document else {}
    return {
        "doc_id": result.doc_id,
        "score": result.score,
        "content": doc.get("content", "") if isinstance(doc, dict) else "",
        **(doc.get("metadata", {}) if isinstance(doc, dict) else {})
    }


class SemanticSearchEngine:
    """
    A semantic search engine using Caliby for:
    - Vector-based semantic search
    - Metadata filtering
    - Faceted navigation
    """
    
    def __init__(self, data_dir: str, embedding_dim: int = EMBEDDING_DIM, reset: bool = False):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        
        caliby.set_buffer_config(size_gb=1.0)
        
        if reset and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        
        caliby.open(data_dir, cleanup_if_exist=reset)
        
        # Schema
        self.schema = caliby.Schema()
        self.schema.add_field("title", caliby.FieldType.STRING)
        self.schema.add_field("author", caliby.FieldType.STRING)
        self.schema.add_field("category", caliby.FieldType.STRING)
        self.schema.add_field("date", caliby.FieldType.STRING)
        self.schema.add_field("views", caliby.FieldType.INT)
        
        try:
            self.documents = caliby.Collection.open("documents")
            print(f"Opened existing index: {self.documents.doc_count()} documents")
        except:
            self.documents = caliby.Collection(
                "documents",
                self.schema,
                vector_dim=embedding_dim,
                distance_metric=caliby.DistanceMetric.COSINE
            )
            self.documents.create_hnsw_index("doc_vectors", M=16, ef_construction=200)
            self.documents.create_metadata_index("category_idx", ["category"])
            self.documents.create_metadata_index("author_idx", ["author"])
            print("Created new search index")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using sentence-transformers."""
        if USE_REAL_EMBEDDINGS and _model:
            try:
                embedding = _model.encode(text, convert_to_numpy=True)
                return embedding.astype(np.float32)
            except Exception as e:
                print(f"Warning: Embedding error: {e}. Using simulated embedding.")
        
        # Fallback: simulated embeddings
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return embedding / np.linalg.norm(embedding)
    
    def index_document(self, title: str, content: str, author: str,
                       category: str, date: str, views: int = 0) -> int:
        """Index a document."""
        metadata = {
            "title": title,
            "author": author,
            "category": category,
            "date": date,
            "views": views
        }
        embedding = self._get_embedding(f"{title} {content}")
        
        ids = self.documents.add([content], [metadata], [embedding.tolist()])
        return ids[0]
    
    def search(self, query: str, k: int = 10,
               category: Optional[str] = None,
               author: Optional[str] = None,
               min_views: Optional[int] = None) -> List[Dict]:
        """Semantic search with optional filters."""
        query_embedding = self._get_embedding(query)
        
        # Build filter
        conditions = {}
        if category:
            conditions["category"] = {"$eq": category}
        if author:
            conditions["author"] = {"$eq": author}
        if min_views:
            conditions["views"] = {"$gte": min_views}
        
        filter_str = json.dumps(conditions) if conditions else ""
        
        results = self.documents.search_vector(
            query_embedding.astype(np.float32),
            "doc_vectors",
            k,
            filter_str
        )
        
        result_dicts = [search_result_to_dict(r) for r in results]
        
        # Add snippet highlighting (simplified)
        query_words = set(query.lower().split())
        for r in result_dicts:
            content = r.get("content", "")
            r["snippet"] = self._create_snippet(content, query_words)
        
        return result_dicts
    
    def _create_snippet(self, content: str, query_words: set, max_len: int = 150) -> str:
        """Create a snippet with query terms highlighted."""
        words = content.split()
        if len(words) <= max_len // 5:
            return content
        
        # Find best starting position
        best_pos = 0
        best_score = 0
        for i in range(len(words)):
            window = words[i:i + max_len // 5]
            score = sum(1 for w in window if w.lower().strip(".,!?") in query_words)
            if score > best_score:
                best_score = score
                best_pos = i
        
        snippet = " ".join(words[best_pos:best_pos + max_len // 5])
        if best_pos > 0:
            snippet = "..." + snippet
        if best_pos + max_len // 5 < len(words):
            snippet = snippet + "..."
        return snippet
    
    def stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_documents": self.documents.doc_count(),
            "embedding_dim": self.embedding_dim
        }
    
    def close(self):
        """Close the engine."""
        caliby.close()


def main():
    print("=" * 70)
    print("Caliby Semantic Search Example")
    print("=" * 70)
    print()
    
    data_dir = "/tmp/caliby_search"
    
    # Initialize
    print("1. Initializing search engine...")
    engine = SemanticSearchEngine(data_dir, embedding_dim=EMBEDDING_DIM, reset=True)
    print("   ✓ Engine ready\n")
    
    # Index sample documents
    print("2. Indexing documents...")
    docs = [
        ("Intro to Machine Learning", 
         "Machine learning enables computers to learn from data without explicit programming.",
         "Alice Smith", "Technology", "2024-01-15", 1500),
        ("Deep Learning Fundamentals",
         "Deep learning uses neural networks with multiple layers to learn representations.",
         "Bob Johnson", "Technology", "2024-02-20", 2300),
        ("Python for Data Science",
         "Python is the most popular language for data science and machine learning.",
         "Alice Smith", "Programming", "2024-01-10", 3200),
        ("Database Design Patterns",
         "Good database design is crucial for application performance and scalability.",
         "Carol Williams", "Programming", "2024-03-05", 890),
        ("Vector Search Explained",
         "Vector search uses embeddings to find semantically similar content.",
         "Bob Johnson", "Technology", "2024-02-28", 1800),
    ]
    
    for title, content, author, category, date, views in docs:
        engine.index_document(title, content, author, category, date, views)
    print(f"   ✓ Indexed {len(docs)} documents\n")
    
    # Basic search
    print("3. Search: 'machine learning algorithms'")
    results = engine.search("machine learning algorithms", k=3)
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        score = r.get("score", 0)
        print(f"   {i}. {title} (score={score:.3f})")
    print()
    
    # Filtered search
    print("4. Search with filter: 'data' in Technology category")
    results = engine.search("data", k=3, category="Technology")
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        category = r.get("category", "")
        print(f"   {i}. {title} [{category}]")
    print()
    
    # Author filter
    print("5. Search by author: content by 'Alice Smith'")
    results = engine.search("programming", k=3, author="Alice Smith")
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        author = r.get("author", "")
        print(f"   {i}. {title} by {author}")
    print()
    
    # Stats
    print("6. Statistics:")
    stats = engine.stats()
    print(f"   Total documents: {stats['total_documents']}")
    print()
    
    # Cleanup
    engine.close()
    shutil.rmtree(data_dir)
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
