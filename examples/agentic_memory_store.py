#!/usr/bin/env python3
"""
Agentic Memory Store Example

This example demonstrates how to use Caliby as a memory store for AI agents.
Agents can store and retrieve memories based on semantic similarity, recency,
importance, and context (conversation, task, etc.).

Use Cases:
- Long-term memory for LLM-based agents
- Episodic memory for chatbots
- Knowledge persistence across sessions
- Context-aware memory retrieval
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import caliby
import numpy as np
import shutil
import time
from datetime import datetime
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


def get_embedding(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Generate embedding using sentence-transformers (all-MiniLM-L6-v2).
    Falls back to simulated embeddings if not available.
    """
    if USE_REAL_EMBEDDINGS and _model:
        try:
            embedding = _model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Warning: Embedding error: {e}. Using simulated embedding.")
    
    # Fallback: simulated embeddings
    np.random.seed(hash(text) % (2**32))
    embedding = np.random.randn(dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


class AgentMemoryStore:
    """
    A memory store for AI agents with support for:
    - Semantic search (find similar memories)
    - Recency filtering (prioritize recent memories)
    - Importance scoring (prioritize important memories)
    - Context tagging (organize by conversation, task, etc.)
    """
    
    def __init__(self, data_dir: str, embedding_dim: int = EMBEDDING_DIM, 
                 buffer_size_gb: float = 1.0, reset: bool = False):
        """
        Initialize the memory store.
        
        Args:
            data_dir: Directory to store memory data
            embedding_dim: Dimension of embedding vectors
            buffer_size_gb: Size of buffer pool in GB
            reset: If True, clear existing data
        """
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        
        # Initialize Caliby
        caliby.set_buffer_config(size_gb=buffer_size_gb)
        
        if reset and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        
        caliby.open(data_dir, cleanup_if_exist=reset)
        
        # Create schema for memories
        self.schema = caliby.Schema()
        self.schema.add_field("memory_type", caliby.FieldType.STRING)  # episodic, semantic, procedural
        self.schema.add_field("context_id", caliby.FieldType.STRING)   # conversation/task ID
        self.schema.add_field("timestamp", caliby.FieldType.FLOAT)     # Unix timestamp
        self.schema.add_field("importance", caliby.FieldType.FLOAT)    # 0.0 to 1.0
        self.schema.add_field("access_count", caliby.FieldType.INT)    # Times retrieved
        self.schema.add_field("tags", caliby.FieldType.STRING_ARRAY)   # Custom tags
        
        # Try to open existing collection or create new one
        try:
            self.collection = caliby.Collection.open("agent_memories")
            print(f"Opened existing memory store with {self.collection.doc_count()} memories")
        except:
            self.collection = caliby.Collection(
                "agent_memories", 
                self.schema, 
                vector_dim=embedding_dim,
                distance_metric=caliby.DistanceMetric.COSINE
            )
            # Create HNSW index for semantic search
            self.collection.create_hnsw_index("memory_vectors", M=16, ef_construction=200)
            # Create metadata index for filtering
            self.collection.create_metadata_index("memory_meta", ["memory_type"])
            print("Created new memory store")
    
    def store_memory(self, content: str, embedding: np.ndarray,
                     memory_type: str = "episodic",
                     context_id: str = "default",
                     importance: float = 0.5,
                     tags: List[str] = None) -> int:
        """Store a new memory."""
        metadata = {
            "memory_type": memory_type,
            "context_id": context_id,
            "timestamp": time.time(),
            "importance": importance,
            "access_count": 0,
            "tags": tags or []
        }
        
        ids = self.collection.add(
            [content],
            [metadata],
            [embedding.tolist()]
        )
        return ids[0]
    
    def recall(self, query_embedding: np.ndarray, k: int = 10,
               memory_type: Optional[str] = None,
               context_id: Optional[str] = None,
               min_importance: float = 0.0,
               recency_weight: float = 0.0) -> List[Dict]:
        """Recall memories similar to the query."""
        import json
        
        # Build filter as JSON string
        filter_conditions = {}
        if memory_type:
            filter_conditions["memory_type"] = {"$eq": memory_type}
        if context_id:
            filter_conditions["context_id"] = {"$eq": context_id}
        if min_importance > 0:
            filter_conditions["importance"] = {"$gte": min_importance}
        
        filter_str = json.dumps(filter_conditions) if filter_conditions else ""
        
        # Search
        results = self.collection.search_vector(
            query_embedding.astype(np.float32),
            "memory_vectors",
            k * 2,  # Fetch more for post-filtering
            filter_str
        )
        
        # Convert SearchResult objects to dictionaries
        result_dicts = []
        for r in results:
            doc = r.document if r.document else {}
            result_dict = {
                "doc_id": r.doc_id,
                "score": r.score,
                "content": doc.get("content", "") if isinstance(doc, dict) else "",
                **(doc.get("metadata", {}) if isinstance(doc, dict) else {})
            }
            result_dicts.append(result_dict)
        
        # Apply recency weighting if specified
        if recency_weight > 0 and result_dicts:
            current_time = time.time()
            for r in result_dicts:
                age_hours = (current_time - r.get("timestamp", current_time)) / 3600
                recency_score = np.exp(-age_hours / 24)  # Decay over 24 hours
                r["final_score"] = (1 - recency_weight) * r["score"] + recency_weight * recency_score
            result_dicts.sort(key=lambda x: x.get("final_score", x["score"]), reverse=True)
        
        return result_dicts[:k]
    
    def stats(self) -> Dict:
        """Get memory store statistics."""
        return {
            "total_memories": self.collection.doc_count(),
            "embedding_dim": self.embedding_dim,
            "data_dir": self.data_dir
        }
    
    def close(self):
        """Close the memory store and flush to disk."""
        caliby.close()


def main():
    print("=" * 70)
    print("Caliby Agentic Memory Store Example")
    print("=" * 70)
    print()
    
    data_dir = "/tmp/caliby_agent_memory"
    
    # Initialize memory store
    print("1. Initializing memory store...")
    memory = AgentMemoryStore(data_dir, embedding_dim=EMBEDDING_DIM, reset=True)
    print(f"   ✓ Memory store ready at {data_dir}\n")
    
    # Simulate a conversation with memory storage
    print("2. Storing conversation memories...")
    conversation_id = "conv_001"
    
    messages = [
        ("User: What's the weather like today?", 0.3),
        ("Agent: I don't have access to real-time weather data, but I can help you find a weather service.", 0.4),
        ("User: I'm planning a trip to Paris next month.", 0.7),
        ("Agent: Paris in the spring is lovely! Would you like recommendations for things to do?", 0.5),
        ("User: Yes, I'm especially interested in art museums.", 0.8),
        ("Agent: The Louvre and Musée d'Orsay are must-visits. The Orangerie has beautiful Monet works.", 0.9),
        ("User: I also love impressionist art.", 0.8),
        ("Agent: Then definitely visit Musée d'Orsay - it has the world's largest impressionist collection.", 0.9),
    ]
    
    for content, importance in messages:
        embedding = get_embedding(content)
        memory_id = memory.store_memory(
            content=content,
            embedding=embedding,
            memory_type="episodic",
            context_id=conversation_id,
            importance=importance,
            tags=["conversation", "travel"] if "Paris" in content or "trip" in content else ["conversation"]
        )
        print(f"   Stored: '{content[:50]}...' (importance={importance})")
    
    print(f"\n   ✓ Stored {len(messages)} memories\n")
    
    # Store some semantic/factual memories
    print("3. Storing semantic (factual) memories...")
    facts = [
        ("The Louvre is the world's most-visited art museum, located in Paris.", ["fact", "paris", "art"]),
        ("Musée d'Orsay is famous for impressionist and post-impressionist masterpieces.", ["fact", "paris", "art"]),
        ("The best time to visit Paris is spring (April-June) or fall (September-November).", ["fact", "paris", "travel"]),
        ("User prefers impressionist art style.", ["user_preference", "art"]),
    ]
    
    for content, tags in facts:
        embedding = get_embedding(content)
        memory.store_memory(
            content=content,
            embedding=embedding,
            memory_type="semantic",
            context_id="global",  # Global knowledge
            importance=0.8,
            tags=tags
        )
        print(f"   Stored: '{content[:50]}...'")
    
    print(f"\n   ✓ Stored {len(facts)} semantic memories\n")
    
    # Recall memories based on a new query
    print("4. Recalling relevant memories for a new query...")
    query = "What should I see in Paris?"
    query_embedding = get_embedding(query)
    
    print(f"   Query: '{query}'")
    print("   Results:")
    
    results = memory.recall(
        query_embedding,
        k=5,
        recency_weight=0.2  # Slightly prefer recent memories
    )
    
    for i, result in enumerate(results, 1):
        content = result.get("content", "")[:60]
        score = result.get("score", 0)
        mem_type = result.get("memory_type", "unknown")
        print(f"   {i}. [{mem_type}] {content}... (score={score:.3f})")
    
    print()
    
    # Recall with filters
    print("5. Filtered recall (only semantic memories)...")
    results = memory.recall(
        query_embedding,
        k=3,
        memory_type="semantic"
    )
    
    for i, result in enumerate(results, 1):
        content = result.get("content", "")[:60]
        print(f"   {i}. {content}...")
    
    print()
    
    # Show statistics
    print("6. Memory store statistics:")
    stats = memory.stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Embedding dimension: {stats['embedding_dim']}")
    print()
    
    # Close memory store
    print("7. Closing memory store...")
    memory.close()
    print("   ✓ Memory store closed and data persisted\n")
    
    # Cleanup
    shutil.rmtree(data_dir)
    print(f"✅ Example complete! Cleaned up {data_dir}")


if __name__ == "__main__":
    main()
