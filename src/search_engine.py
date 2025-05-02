"""
Search Engine - Handles semantic search functionality for fashion images

This module:
1. Loads a pre-built FAISS index and associated metadata
2. Provides functions for text-based search (with SentenceTransformers)
3. (Optional) Provides functions for image-based search (with CLIP)

Usage:
    from src.search_engine import FashionSearchEngine
    
    # Initialize search engine
    search_engine = FashionSearchEngine('data/captions.index', 'data/image_paths.json')
    
    # Search by text
    results = search_engine.search_by_text("red floral summer dress", top_k=5)
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class FashionSearchEngine:
    """Fashion image search engine using FAISS and SentenceTransformers"""
    
    def __init__(self, index_path, image_paths_file, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the search engine with a pre-built index.
        
        Args:
            index_path: Path to the FAISS index file
            image_paths_file: Path to the JSON file with image paths
            model_name: Name of the SentenceTransformer model for text embedding
        """
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load image paths
        with open(image_paths_file, 'r') as f:
            self.image_paths = json.load(f)
        print(f"Loaded {len(self.image_paths)} image paths")
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        print(f"Initialized SentenceTransformer model: {model_name}")
        
        # Initialize CLIP model for image search (only if needed)
        self.clip_model = None
        self.clip_processor = None
    
    def search_by_text(self, query_text, top_k=5):
        """
        Search for images similar to the text query.
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with image paths, captions, and similarity scores
        """
        # Embed the query text
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        
        # Normalize the embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append({
                    'image_path': self.image_paths[idx],
                    'similarity': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def initialize_clip(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model for image-based search (only when needed).
        
        Args:
            model_name: Name of the CLIP model to use
        """
        if self.clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                
                print(f"Initializing CLIP model for image search: {model_name}")
                self.clip_model = CLIPModel.from_pretrained(model_name)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name)
                
                # Use GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model = self.clip_model.to(device)
                self.device = device
                
                print(f"CLIP model loaded successfully. Using device: {device}")
            except ImportError:
                print("Error: transformers package with CLIP support is required for image search")
                raise
    
    def search_by_image(self, image_path, top_k=5):
        """
        Search for images similar to the query image.
        
        Args:
            image_path: Path to the query image
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with image paths and similarity scores
        """
        # Initialize CLIP if not already done
        self.initialize_clip()
        
        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_embedding = image_features.cpu().numpy()
        
        # Normalize the embedding
        faiss.normalize_L2(image_embedding)
        
        # Search the index
        scores, indices = self.index.search(image_embedding.astype(np.float32), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append({
                    'image_path': self.image_paths[idx],
                    'similarity': float(score),
                    'rank': i + 1
                })
        
        return results
    
    def get_captions_for_results(self, results, captions_file):
        """
        Add caption information to search results.
        
        Args:
            results: List of search result dictionaries
            captions_file: Path to the processed captions JSON file
            
        Returns:
            Updated results with captions
        """
        # Load captions
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        
        # Create a mapping from image path to caption
        path_to_caption = {item['image_path']: item['caption'] for item in captions_data}
        
        # Add captions to results
        for result in results:
            image_path = result['image_path']
            if image_path in path_to_caption:
                result['caption'] = path_to_caption[image_path]
        
        return results

# Simple command-line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Search for fashion images')
    parser.add_argument('--index_path', type=str, default='data/captions.index',
                        help='Path to the FAISS index file')
    parser.add_argument('--image_paths_file', type=str, default='data/image_paths.json',
                        help='Path to the JSON file with image paths')
    parser.add_argument('--captions_file', type=str, default='data/processed_captions.json',
                        help='Path to the processed captions JSON file')
    parser.add_argument('--query', type=str, required=True,
                        help='Text query to search for')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top results to return')
    
    args = parser.parse_args()
    
    # Initialize search engine
    search_engine = FashionSearchEngine(args.index_path, args.image_paths_file)
    
    # Search by text
    results = search_engine.search_by_text(args.query, args.top_k)
    
    # Add captions if available
    if os.path.exists(args.captions_file):
        results = search_engine.get_captions_for_results(results, args.captions_file)
    
    # Print results
    print(f"\nTop {len(results)} results for query: '{args.query}'")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"Result #{i+1} (Score: {result['similarity']:.4f}):")
        print(f"  Image: {result['image_path']}")
        if 'caption' in result:
            print(f"  Caption: {result['caption']}")
        print()