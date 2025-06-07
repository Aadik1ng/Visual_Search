import faiss
import numpy as np
import json
import os
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time

class VisualSearchEngine:
    def __init__(self, data_dir="data", model_name="openai/clip-vit-base-patch32"):
        """Initialize the visual search engine with FAISS index"""
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP model for query processing
        print("Loading CLIP model for query processing...")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Load embeddings and metadata
        self.embeddings = None
        self.product_ids = None
        self.products_data = None
        self.faiss_index = None
        
        self._load_data()
        self._build_faiss_index()
        
        print("Visual search engine initialized successfully!")
    
    def _load_data(self):
        """Load embeddings, product IDs, and product data"""
        embeddings_dir = os.path.join(self.data_dir, "embeddings")
        processed_dir = os.path.join(self.data_dir, "processed")
        
        # Load embeddings
        embeddings_file = os.path.join(embeddings_dir, "embeddings.npy")
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        self.embeddings = np.load(embeddings_file)
        print(f"Loaded {self.embeddings.shape[0]} embeddings of dimension {self.embeddings.shape[1]}")
        
        # Load product IDs
        product_ids_file = os.path.join(embeddings_dir, "product_ids.json")
        with open(product_ids_file, 'r') as f:
            self.product_ids = json.load(f)
        
        # Load product data
        products_file = os.path.join(processed_dir, "products_with_embeddings.json")
        with open(products_file, 'r', encoding='utf-8') as f:
            products_list = json.load(f)
        
        # Create product lookup dictionary
        self.products_data = {product['product_id']: product for product in products_list}
        print(f"Loaded data for {len(self.products_data)} products")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")
        
        # Ensure embeddings are float32 and normalized
        embeddings_normalized = self.embeddings.astype('float32')
        
        # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
        dimension = embeddings_normalized.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_normalized)
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def _extract_query_embedding(self, image):
        """Extract CLIP embedding for query image"""
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = F.normalize(image_features, p=2, dim=1)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            return None
    
    def search_by_image(self, image, top_k=10) -> List[dict]:
        """Search for similar products using an image"""
        start_time = time.time()
        
        # Extract query embedding
        query_embedding = self._extract_query_embedding(image)
        if query_embedding is None:
            return []
        
        # Search using FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                product_data = self.products_data.get(product_id, {})
                
                result = {
                    'rank': i + 1,
                    'product_id': product_id,
                    'similarity_score': float(similarity),
                    'product_name': product_data.get('product_name_clean', ''),
                    'brand': product_data.get('brand_clean', ''),
                    'selling_price': product_data.get('selling_price_clean', 0),
                    'mrp': product_data.get('mrp_clean', 0),
                    'category_id': product_data.get('category_id', ''),
                    'description': product_data.get('description_clean', ''),
                    'image_url': product_data.get('feature_image_s3', ''),
                    'pdp_url': product_data.get('pdp_url', ''),
                    'features': product_data.get('features_parsed', [])
                }
                results.append(result)
        
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.3f} seconds")
        
        return results
    
    def search_by_image_path(self, image_path, top_k=10) -> List[dict]:
        """Search for similar products using an image file path"""
        try:
            image = Image.open(image_path)
            return self.search_by_image(image, top_k)
        except Exception as e:
            print(f"Error loading image from path {image_path}: {e}")
            return []
    
    def search_by_product_id(self, product_id, top_k=10) -> List[dict]:
        """Search for similar products using a product ID"""
        try:
            # Find the index of the product
            if product_id not in self.product_ids:
                print(f"Product ID {product_id} not found")
                return []
            
            product_idx = self.product_ids.index(product_id)
            
            # Get the embedding
            query_embedding = self.embeddings[product_idx].reshape(1, -1).astype('float32')
            
            # Search using FAISS
            similarities, indices = self.faiss_index.search(query_embedding, top_k + 1)  # +1 to exclude self
            
            # Prepare results (skip the first one which is the query itself)
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0][1:], indices[0][1:])):
                if idx < len(self.product_ids):
                    similar_product_id = self.product_ids[idx]
                    product_data = self.products_data.get(similar_product_id, {})
                    
                    result = {
                        'rank': i + 1,
                        'product_id': similar_product_id,
                        'similarity_score': float(similarity),
                        'product_name': product_data.get('product_name_clean', ''),
                        'brand': product_data.get('brand_clean', ''),
                        'selling_price': product_data.get('selling_price_clean', 0),
                        'mrp': product_data.get('mrp_clean', 0),
                        'category_id': product_data.get('category_id', ''),
                        'description': product_data.get('description_clean', ''),
                        'image_url': product_data.get('feature_image_s3', ''),
                        'pdp_url': product_data.get('pdp_url', ''),
                        'features': product_data.get('features_parsed', [])
                    }
                    results.append(result)
            
            return results
        
        except Exception as e:
            print(f"Error searching by product ID {product_id}: {e}")
            return []
    
    def get_product_info(self, product_id) -> Optional[dict]:
        """Get detailed information for a specific product"""
        return self.products_data.get(product_id)
    
    def get_random_products(self, num_products=10) -> List[dict]:
        """Get random products for testing/demo purposes"""
        import random
        
        random_ids = random.sample(self.product_ids, min(num_products, len(self.product_ids)))
        
        results = []
        for product_id in random_ids:
            product_data = self.products_data.get(product_id, {})
            result = {
                'product_id': product_id,
                'product_name': product_data.get('product_name_clean', ''),
                'brand': product_data.get('brand_clean', ''),
                'selling_price': product_data.get('selling_price_clean', 0),
                'image_url': product_data.get('feature_image_s3', ''),
                'category_id': product_data.get('category_id', '')
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> dict:
        """Get search engine statistics"""
        return {
            'total_products': len(self.product_ids),
            'embedding_dimension': self.embeddings.shape[1],
            'model_name': self.model_name,
            'device': self.device,
            'index_type': 'FAISS IndexFlatIP'
        }

def test_search_engine():
    """Test function for the search engine"""
    print("Testing Visual Search Engine...")
    
    # Initialize search engine
    engine = VisualSearchEngine()
    
    # Print stats
    stats = engine.get_stats()
    print(f"Search engine stats: {stats}")
    
    # Test with random products
    print("\nTesting with random products...")
    random_products = engine.get_random_products(3)
    
    for product in random_products:
        print(f"\nTesting with product: {product['product_name']} (ID: {product['product_id']})")
        
        # Search for similar products
        similar_products = engine.search_by_product_id(product['product_id'], top_k=5)
        
        print("Similar products found:")
        for result in similar_products:
            print(f"  {result['rank']}. {result['product_name']} (Score: {result['similarity_score']:.3f})")
    
    print("\nSearch engine test completed!")

if __name__ == "__main__":
    test_search_engine() 