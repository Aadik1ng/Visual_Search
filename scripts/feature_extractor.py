import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import time

class CLIPFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """Initialize CLIP model for feature extraction"""
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        print("CLIP model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def extract_single_embedding(self, image_path):
        """Extract CLIP embedding for a single image"""
        image = self.preprocess_image(image_path)
        if image is None:
            return None
        
        try:
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings for cosine similarity
                image_features = F.normalize(image_features, p=2, dim=1)
            
            return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error extracting features for {image_path}: {e}")
            return None
    
    def extract_batch_embeddings(self, image_paths, batch_size=32):
        """Extract embeddings for a batch of images"""
        embeddings = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # Load batch images
            for path in batch_paths:
                image = self.preprocess_image(path)
                if image is not None:
                    batch_images.append(image)
                    batch_valid_paths.append(path)
            
            if not batch_images:
                continue
            
            try:
                # Process batch
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                with torch.no_grad():
                    batch_features = self.model.get_image_features(**inputs)
                    # Normalize embeddings
                    batch_features = F.normalize(batch_features, p=2, dim=1)
                
                # Add to results
                embeddings.extend(batch_features.cpu().numpy())
                valid_paths.extend(batch_valid_paths)
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Fall back to individual processing for this batch
                for path in batch_valid_paths:
                    embedding = self.extract_single_embedding(path)
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_paths.append(path)
        
        return np.array(embeddings), valid_paths
    
    def process_dataset(self, data_dir="data", batch_size=32):
        """Process all images in the dataset"""
        images_dir = os.path.join(data_dir, "images")
        processed_dir = os.path.join(data_dir, "processed")
        embeddings_dir = os.path.join(data_dir, "embeddings")
        
        # Create embeddings directory
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Load product data
        products_file = os.path.join(processed_dir, "products.json")
        if not os.path.exists(products_file):
            raise FileNotFoundError(f"Products file not found: {products_file}")
        
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        print(f"Processing {len(products)} products...")
        
        # Get image paths
        image_paths = []
        product_ids = []
        
        for product in products:
            product_id = product['product_id']
            image_path = os.path.join(images_dir, f"{product_id}.jpg")
            
            if os.path.exists(image_path):
                image_paths.append(image_path)
                product_ids.append(product_id)
            else:
                print(f"Warning: Image not found for product {product_id}")
        
        print(f"Found {len(image_paths)} images to process")
        
        # Extract embeddings
        start_time = time.time()
        embeddings, valid_paths = self.extract_batch_embeddings(image_paths, batch_size)
        end_time = time.time()
        
        print(f"Extracted {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
        print(f"Average time per image: {(end_time - start_time) / len(embeddings):.3f} seconds")
        
        # Create mapping from valid paths to product IDs
        valid_product_ids = []
        for path in valid_paths:
            filename = os.path.basename(path)
            product_id = filename.replace('.jpg', '')
            valid_product_ids.append(product_id)
        
        # Save embeddings
        embeddings_file = os.path.join(embeddings_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings)
        print(f"Embeddings saved to {embeddings_file}")
        
        # Save product ID mapping
        mapping_file = os.path.join(embeddings_dir, "product_ids.json")
        with open(mapping_file, 'w') as f:
            json.dump(valid_product_ids, f, indent=2)
        print(f"Product ID mapping saved to {mapping_file}")
        
        # Save embedding metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": embeddings.shape[1],
            "num_embeddings": len(embeddings),
            "extraction_time": end_time - start_time,
            "device_used": self.device,
            "batch_size": batch_size
        }
        
        metadata_file = os.path.join(embeddings_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_file}")
        
        # Filter products to only include those with embeddings
        products_with_embeddings = [p for p in products if p['product_id'] in valid_product_ids]
        
        # Save filtered products
        filtered_products_file = os.path.join(processed_dir, "products_with_embeddings.json")
        with open(filtered_products_file, 'w', encoding='utf-8') as f:
            json.dump(products_with_embeddings, f, indent=2, ensure_ascii=False)
        print(f"Filtered products saved to {filtered_products_file}")
        
        return embeddings, valid_product_ids, metadata

def main():
    """Main function to extract features from the dataset"""
    print("Starting CLIP feature extraction...")
    
    # Initialize extractor
    extractor = CLIPFeatureExtractor()
    
    # Process dataset
    embeddings, product_ids, metadata = extractor.process_dataset(batch_size=16)
    
    print("\nFeature extraction complete!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Model used: {metadata['model_name']}")

if __name__ == "__main__":
    main() 