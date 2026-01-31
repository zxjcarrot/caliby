#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Pipeline Example

This example demonstrates how to use Caliby for RAG applications with:
- Document chunking and ingestion
- Semantic similarity search
- Context retrieval for LLM augmentation

Use Cases:
- Question-answering over documents
- Chatbots with knowledge bases
- Research assistants
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import caliby
import numpy as np
import shutil
import json
from typing import List, Dict, Optional, Tuple

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


class RAGPipeline:
    """
    A RAG pipeline using Caliby for document storage and retrieval.
    """
    
    def __init__(self, data_dir: str, embedding_dim: int = EMBEDDING_DIM, 
                 chunk_size: int = 500, chunk_overlap: int = 50,
                 reset: bool = False):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        caliby.set_buffer_config(size_gb=1.0)
        
        if reset and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        
        caliby.open(data_dir, cleanup_if_exist=reset)
        
        # Schema for document chunks
        self.schema = caliby.Schema()
        self.schema.add_field("doc_id", caliby.FieldType.STRING)
        self.schema.add_field("source", caliby.FieldType.STRING)
        self.schema.add_field("chunk_index", caliby.FieldType.INT)
        self.schema.add_field("total_chunks", caliby.FieldType.INT)
        
        try:
            self.chunks = caliby.Collection.open("chunks")
            print(f"Opened existing index: {self.chunks.doc_count()} chunks")
        except:
            self.chunks = caliby.Collection(
                "chunks",
                self.schema,
                vector_dim=embedding_dim,
                distance_metric=caliby.DistanceMetric.COSINE
            )
            self.chunks.create_hnsw_index("chunk_vectors", M=16, ef_construction=200)
            self.chunks.create_metadata_index("source_idx", ["source"])
            print("Created new RAG index")
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        return chunks
    
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
    
    def ingest_document(self, doc_id: str, text: str, source: str = "unknown") -> int:
        """Ingest a document by chunking and storing it."""
        chunks = self._split_text(text)
        if not chunks:
            return 0
        
        contents = []
        metadatas = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            contents.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            embeddings.append(self._get_embedding(chunk).tolist())
        
        self.chunks.add(contents, metadatas, embeddings)
        return len(chunks)
    
    def retrieve(self, query: str, k: int = 5, 
                source: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self._get_embedding(query)
        
        filter_str = json.dumps({"source": {"$eq": source}}) if source else ""
        
        results = self.chunks.search_vector(
            query_embedding.astype(np.float32),
            "chunk_vectors",
            k,
            filter_str
        )
        
        return [search_result_to_dict(r) for r in results]
    
    def build_context(self, results: List[Dict], max_tokens: int = 2000) -> str:
        """Build context string from retrieved results."""
        context_parts = []
        token_count = 0
        
        for r in results:
            chunk_text = r.get("content", "")
            chunk_tokens = len(chunk_text.split())
            
            if token_count + chunk_tokens > max_tokens:
                break
            
            source = r.get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{chunk_text}")
            token_count += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def query(self, question: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Execute a RAG query and return context + sources."""
        results = self.retrieve(question, k=k)
        context = self.build_context(results)
        return context, results
    
    def stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "total_chunks": self.chunks.doc_count(),
            "chunk_size": self.chunk_size,
            "embedding_dim": self.embedding_dim
        }
    
    def close(self):
        """Close the pipeline."""
        caliby.close()


def main():
    print("=" * 70)
    print("Caliby RAG Pipeline Example")
    print("=" * 70)
    print()
    
    data_dir = "/tmp/caliby_rag"
    
    # Initialize
    print("1. Initializing RAG pipeline...")
    rag = RAGPipeline(
        data_dir,
        embedding_dim=EMBEDDING_DIM,
        chunk_size=200,
        chunk_overlap=30,
        reset=True
    )
    print("   ✓ Pipeline ready\n")
    
    # Sample documents
    docs = {
        "ml_intro": {
            "source": "machine_learning_guide",
            "text": """
            Machine learning is a subset of artificial intelligence that enables 
            systems to learn from data. It includes supervised learning, unsupervised 
            learning, and reinforcement learning. Neural networks are a key component,
            inspired by the human brain's structure.
            """
        },
        "python_basics": {
            "source": "python_tutorial",
            "text": """
            Python is a high-level programming language known for its readability.
            It supports multiple paradigms: procedural, object-oriented, and functional.
            Python is widely used in data science, web development, and automation.
            """
        },
        "vector_db": {
            "source": "database_overview",
            "text": """
            Vector databases store high-dimensional embeddings for similarity search.
            They use algorithms like HNSW and IVF for approximate nearest neighbor search.
            Common use cases include recommendation systems and semantic search.
            """
        }
    }
    
    # Ingest documents
    print("2. Ingesting documents...")
    for doc_id, doc_data in docs.items():
        chunks = rag.ingest_document(doc_id, doc_data["text"], doc_data["source"])
        print(f"   '{doc_id}': {chunks} chunks")
    print()
    
    # Query
    print("3. RAG Query: 'What is machine learning?'")
    context, sources = rag.query("What is machine learning?", k=3)
    print(f"\n   Retrieved Context:\n   {'-'*50}")
    print(f"   {context[:300]}...")
    print(f"\n   Sources:")
    for s in sources[:3]:
        print(f"   - {s.get('source')} (chunk {s.get('chunk_index')}, score={s.get('score', 0):.3f})")
    print()
    
    # Query with filter
    print("4. Query with source filter: 'programming' in python_tutorial")
    results = rag.retrieve("programming basics", k=3, source="python_tutorial")
    for r in results:
        content = r.get("content", "")[:80]
        print(f"   [{r.get('source')}] {content}...")
    print()
    
    # Stats
    print("5. Pipeline Statistics:")
    stats = rag.stats()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Chunk size: {stats['chunk_size']} chars")
    print()
    
    # Cleanup
    rag.close()
    shutil.rmtree(data_dir)
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
