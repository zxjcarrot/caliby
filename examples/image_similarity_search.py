#!/usr/bin/env python3
"""
Image Similarity Search Example

This example demonstrates how to use Caliby for image similarity search with:
- Image embedding storage
- Visual similarity search
- Reverse image lookup

Note: This example uses text descriptions to generate embeddings.
In production with real images, use CLIP, ResNet, or other vision models.

Use Cases:
- Visual product search
- Duplicate image detection
- Similar image recommendations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import caliby
import numpy as np
import shutil
import json
from typing import List, Dict, Optional

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


def get_image_embedding(image_desc: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """
    Generate embedding for image description using sentence-transformers.
    In production with real images, use CLIP or similar multimodal models.
    """
    if USE_REAL_EMBEDDINGS and _model:
        try:
            embedding = _model.encode(image_desc, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Warning: Embedding error: {e}. Using simulated embedding.")
    
    # Fallback: simulated embeddings
    np.random.seed(hash(image_desc) % (2**32))
    embedding = np.random.randn(dim).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


class ImageSimilaritySearch:
    """
    An image similarity search engine using Caliby.
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
        self.schema.add_field("image_id", caliby.FieldType.STRING)
        self.schema.add_field("filename", caliby.FieldType.STRING)
        self.schema.add_field("category", caliby.FieldType.STRING)
        self.schema.add_field("width", caliby.FieldType.INT)
        self.schema.add_field("height", caliby.FieldType.INT)
        
        try:
            self.images = caliby.Collection.open("images")
            print(f"Opened existing gallery: {self.images.doc_count()} images")
        except:
            self.images = caliby.Collection(
                "images",
                self.schema,
                vector_dim=embedding_dim,
                distance_metric=caliby.DistanceMetric.COSINE
            )
            self.images.create_hnsw_index("image_vectors", M=16, ef_construction=200)
            self.images.create_metadata_index("category_idx", ["category"])
            print("Created new image gallery")
    
    def add_image(self, image_id: str, filename: str, embedding: np.ndarray,
                  category: str = "general", width: int = 0, height: int = 0) -> int:
        """Add an image to the gallery."""
        metadata = {
            "image_id": image_id,
            "filename": filename,
            "category": category,
            "width": width,
            "height": height
        }
        ids = self.images.add(
            [f"Image: {filename}"],
            [metadata],
            [embedding.tolist()]
        )
        return ids[0]
    
    def find_similar(self, query_embedding: np.ndarray, k: int = 10,
                    category: Optional[str] = None,
                    exclude_ids: List[str] = None) -> List[Dict]:
        """Find similar images."""
        exclude_ids = exclude_ids or []
        
        filter_str = json.dumps({"category": {"$eq": category}}) if category else ""
        
        results = self.images.search_vector(
            query_embedding.astype(np.float32),
            "image_vectors",
            k * 2 if exclude_ids else k,
            filter_str
        )
        
        result_dicts = [search_result_to_dict(r) for r in results]
        
        # Filter out excluded
        if exclude_ids:
            result_dicts = [r for r in result_dicts 
                          if r.get("image_id") not in exclude_ids]
        
        return result_dicts[:k]
    
    def reverse_image_search(self, query_embedding: np.ndarray, 
                            threshold: float = 0.95) -> Optional[Dict]:
        """Find exact or near-duplicate images."""
        results = self.find_similar(query_embedding, k=1)
        
        if results and results[0].get("score", 0) >= threshold:
            return results[0]
        return None
    
    def stats(self) -> Dict:
        """Get gallery statistics."""
        return {
            "total_images": self.images.doc_count(),
            "embedding_dim": self.embedding_dim
        }
    
    def close(self):
        """Close the gallery."""
        caliby.close()


def main():
    print("=" * 70)
    print("Caliby Image Similarity Search Example")
    print("=" * 70)
    print()
    
    data_dir = "/tmp/caliby_images"
    
    # Initialize
    print("1. Initializing image gallery...")
    gallery = ImageSimilaritySearch(data_dir, embedding_dim=EMBEDDING_DIM, reset=True)
    print("   ✓ Gallery ready\n")
    
    # Add sample images
    print("2. Adding images to gallery...")
    images = [
        ("IMG001", "sunset_beach.jpg", "nature", "orange sunset over ocean waves", 1920, 1080),
        ("IMG002", "mountain_lake.jpg", "nature", "calm lake with mountain reflection", 2048, 1365),
        ("IMG003", "city_night.jpg", "urban", "cityscape with lights at night", 1920, 1080),
        ("IMG004", "forest_path.jpg", "nature", "path through green forest", 1600, 1200),
        ("IMG005", "cat_portrait.jpg", "animals", "close up of tabby cat face", 1200, 1200),
        ("IMG006", "dog_running.jpg", "animals", "golden retriever running in park", 1920, 1080),
        ("IMG007", "sunset_desert.jpg", "nature", "orange sunset over sand dunes", 2048, 1365),
        ("IMG008", "office_building.jpg", "urban", "modern glass office building", 1600, 1200),
    ]
    
    for img_id, filename, category, desc, w, h in images:
        embedding = get_image_embedding(desc)
        gallery.add_image(img_id, filename, embedding, category, w, h)
    print(f"   ✓ Added {len(images)} images\n")
    
    # Find similar to sunset
    print("3. Find images similar to 'sunset_beach.jpg'...")
    query_embedding = get_image_embedding("orange sunset over ocean waves")
    similar = gallery.find_similar(query_embedding, k=3, exclude_ids=["IMG001"])
    
    for i, r in enumerate(similar, 1):
        filename = r.get("filename", "Unknown")
        score = r.get("score", 0)
        category = r.get("category", "")
        print(f"   {i}. {filename} [{category}] (similarity={score:.3f})")
    print()
    
    # Filter by category
    print("4. Similar nature images to 'mountain lake'...")
    query_embedding = get_image_embedding("calm lake with mountain reflection")
    similar = gallery.find_similar(query_embedding, k=3, category="nature", 
                                   exclude_ids=["IMG002"])
    
    for i, r in enumerate(similar, 1):
        filename = r.get("filename", "Unknown")
        score = r.get("score", 0)
        print(f"   {i}. {filename} (similarity={score:.3f})")
    print()
    
    # Reverse image search
    print("5. Reverse image search (find duplicates)...")
    
    # Search for exact match
    exact_embedding = get_image_embedding("close up of tabby cat face")
    match = gallery.reverse_image_search(exact_embedding, threshold=0.95)
    if match:
        print(f"   ✓ Found match: {match.get('filename')} (score={match.get('score', 0):.3f})")
    else:
        print("   No exact match found")
    
    # Search for non-match
    new_embedding = get_image_embedding("completely different image")
    match = gallery.reverse_image_search(new_embedding, threshold=0.95)
    if match:
        print(f"   Found: {match.get('filename')}")
    else:
        print("   ✓ No match for new image (expected)")
    print()
    
    # Stats
    print("6. Gallery Statistics:")
    stats = gallery.stats()
    print(f"   Total images: {stats['total_images']}")
    print(f"   Embedding dimension: {stats['embedding_dim']}")
    print()
    
    # Cleanup
    gallery.close()
    shutil.rmtree(data_dir)
    print("✅ Example complete!")


if __name__ == "__main__":
    main()
