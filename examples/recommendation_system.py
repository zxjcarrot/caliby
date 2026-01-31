#!/usr/bin/env python3
"""
Recommendation System Example

This example demonstrates how to use Caliby to build a product recommendation
system with content-based filtering and personalized suggestions.

Use Cases:
- E-commerce product recommendations
- Content recommendation (articles, videos)
- "You might also like" features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import caliby
import numpy as np
import shutil
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

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
    return embedding / np.linalg.norm(embedding)


@dataclass
class Product:
    """Product data structure."""
    id: str
    name: str
    category: str
    price: float
    description: str
    popularity: float = 0.0


class RecommendationEngine:
    """
    A product recommendation engine using Caliby for:
    - Content-based similarity search
    - Category filtering
    - Popularity boosting
    """
    
    def __init__(self, data_dir: str, embedding_dim: int = EMBEDDING_DIM,
                 buffer_size_gb: float = 1.0, reset: bool = False):
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        
        caliby.set_buffer_config(size_gb=buffer_size_gb)
        
        if reset and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        
        caliby.open(data_dir, cleanup_if_exist=reset)
        
        # Product schema
        self.schema = caliby.Schema()
        self.schema.add_field("product_id", caliby.FieldType.STRING)
        self.schema.add_field("name", caliby.FieldType.STRING)
        self.schema.add_field("category", caliby.FieldType.STRING)
        self.schema.add_field("price", caliby.FieldType.FLOAT)
        self.schema.add_field("popularity", caliby.FieldType.FLOAT)
        
        try:
            self.products = caliby.Collection.open("products")
            print(f"Opened existing catalog: {self.products.doc_count()} products")
        except:
            self.products = caliby.Collection(
                "products",
                self.schema,
                vector_dim=embedding_dim,
                distance_metric=caliby.DistanceMetric.COSINE
            )
            self.products.create_hnsw_index("product_vectors", M=16, ef_construction=200)
            self.products.create_metadata_index("category_idx", ["category"])
            print("Created new product catalog")
    
    def add_product(self, product: Product, embedding: np.ndarray) -> int:
        """Add a product to the catalog."""
        metadata = {
            "product_id": product.id,
            "name": product.name,
            "category": product.category,
            "price": product.price,
            "popularity": product.popularity
        }
        ids = self.products.add(
            [product.description],
            [metadata],
            [embedding.tolist()]
        )
        return ids[0]
    
    def recommend(self, query_embedding: np.ndarray, k: int = 10,
                 category: Optional[str] = None,
                 max_price: Optional[float] = None,
                 exclude_ids: List[str] = None,
                 popularity_boost: float = 0.2) -> List[Dict]:
        """Get product recommendations."""
        exclude_ids = exclude_ids or []
        
        # Build filter
        filter_conditions = {}
        if category:
            filter_conditions["category"] = {"$eq": category}
        if max_price:
            filter_conditions["price"] = {"$lte": max_price}
        
        filter_str = json.dumps(filter_conditions) if filter_conditions else ""
        
        # Search
        results = self.products.search_vector(
            query_embedding.astype(np.float32),
            "product_vectors",
            k * 2,
            filter_str
        )
        
        # Convert to dicts and filter
        result_dicts = [search_result_to_dict(r) for r in results]
        result_dicts = [r for r in result_dicts if r.get("product_id") not in exclude_ids]
        
        # Apply popularity boost
        if popularity_boost > 0:
            for r in result_dicts:
                similarity = r.get("score", 0)
                pop = r.get("popularity", 0)
                r["final_score"] = (1 - popularity_boost) * similarity + popularity_boost * pop
            result_dicts.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return result_dicts[:k]
    
    def stats(self) -> Dict:
        """Get catalog statistics."""
        return {
            "total_products": self.products.doc_count(),
            "embedding_dim": self.embedding_dim
        }
    
    def close(self):
        """Close the engine."""
        caliby.close()


def main():
    print("=" * 70)
    print("Caliby Recommendation System Example")
    print("=" * 70)
    print()
    
    data_dir = "/tmp/caliby_recs"
    
    # Initialize
    print("1. Initializing recommendation engine...")
    engine = RecommendationEngine(data_dir, embedding_dim=EMBEDDING_DIM, reset=True)
    print("   ✓ Engine ready\n")
    
    # Add products
    print("2. Adding products...")
    products = [
        Product("E001", "Wireless Headphones Pro", "Electronics", 299.99, 
                "Premium noise-canceling wireless headphones", 0.92),
        Product("E002", "Budget Earbuds", "Electronics", 29.99,
                "Affordable wired earbuds with good sound", 0.75),
        Product("E003", "Smart Watch Ultra", "Electronics", 499.99,
                "Advanced smartwatch with health monitoring", 0.88),
        Product("E004", "Portable Charger", "Electronics", 49.99,
                "20000mAh portable battery pack", 0.82),
        Product("B001", "Python Mastery", "Books", 45.99,
                "Complete guide to Python programming", 0.85),
        Product("B002", "AI Revolution", "Books", 32.99,
                "Exploring the future of artificial intelligence", 0.78),
        Product("S001", "Running Shoes Elite", "Sports", 159.99,
                "Lightweight running shoes with cushioning", 0.90),
        Product("S002", "Yoga Mat Pro", "Sports", 79.99,
                "Extra thick yoga mat with anti-slip surface", 0.72),
    ]
    
    for p in products:
        embedding = get_embedding(f"{p.name} {p.description} {p.category}")
        engine.add_product(p, embedding)
    print(f"   ✓ Added {len(products)} products\n")
    
    # Get recommendations
    print("3. Personalized recommendations (tech enthusiast)...")
    user_embedding = get_embedding("technology gadgets electronics audio")
    recs = engine.recommend(user_embedding, k=5, popularity_boost=0.3)
    
    for i, r in enumerate(recs, 1):
        name = r.get("name", "Unknown")
        category = r.get("category", "")
        price = r.get("price", 0)
        print(f"   {i}. {name} [{category}] - ${price:.2f}")
    print()
    
    # Category filter
    print("4. Electronics under $100...")
    affordable = engine.recommend(
        user_embedding, k=5,
        category="Electronics",
        max_price=100.0
    )
    for i, r in enumerate(affordable, 1):
        name = r.get("name", "Unknown")
        price = r.get("price", 0)
        print(f"   {i}. {name} - ${price:.2f}")
    print()
    
    # Stats
    print("5. Statistics:")
    stats = engine.stats()
    print(f"   Products: {stats['total_products']}")
    print()
    
    # Cleanup
    engine.close()
    shutil.rmtree(data_dir)
    print(f"✅ Example complete!")


if __name__ == "__main__":
    main()
