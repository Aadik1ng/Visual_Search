import pandas as pd
import json
import ast
import os
import requests
from urllib.parse import urlparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import hashlib

class DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_datasets(self, csv_files):
        """Load and combine multiple CSV files"""
        print("Loading datasets...")
        dataframes = []
        
        for csv_file in csv_files:
            print(f"Loading {csv_file}...")
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        
        # Combine all datasets
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total products loaded: {len(combined_df)}")
        return combined_df
    
    def clean_data(self, df):
        """Clean and normalize the dataset"""
        print("Cleaning data...")
        
        # Parse price fields (they're stored as string dictionaries)
        def parse_price(price_str):
            try:
                if isinstance(price_str, str):
                    price_dict = ast.literal_eval(price_str)
                    # Get USD price or first available price
                    if 'USD' in price_dict:
                        return float(price_dict['USD'])
                    else:
                        return float(list(price_dict.values())[0])
                return float(price_str) if price_str else 0.0
            except:
                return 0.0
        
        # Clean selling price and MRP
        df['selling_price_clean'] = df['selling_price'].apply(parse_price)
        df['mrp_clean'] = df['mrp'].apply(parse_price)
        
        # Parse feature list
        def parse_feature_list(feature_str):
            try:
                if isinstance(feature_str, str):
                    return ast.literal_eval(feature_str)
                return []
            except:
                return []
        
        df['features_parsed'] = df['feature_list'].apply(parse_feature_list)
        
        # Parse additional images
        def parse_image_list(img_str):
            try:
                if isinstance(img_str, str):
                    return ast.literal_eval(img_str)
                return []
            except:
                return []
        
        df['additional_images'] = df['pdp_images_s3'].apply(parse_image_list)
        
        # Clean text fields
        df['product_name_clean'] = df['product_name'].fillna('').astype(str)
        df['brand_clean'] = df['brand'].fillna('').astype(str)
        df['description_clean'] = df['description'].fillna('').astype(str)
        
        # Remove rows with missing essential data
        df_clean = df.dropna(subset=['product_id', 'feature_image_s3'])
        
        print(f"Cleaned dataset size: {len(df_clean)}")
        return df_clean
    
    def download_image(self, url, product_id, max_retries=3):
        """Download a single image with retry logic"""
        filename = f"{product_id}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            return filepath
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                # Verify it's an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    print(f"Warning: {url} is not an image (content-type: {content_type})")
                    return None
                
                # Save image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify image can be opened
                try:
                    with Image.open(filepath) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(filepath, 'JPEG', quality=85)
                    return filepath
                except Exception as e:
                    print(f"Error processing image {filepath}: {e}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return None
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def download_images(self, df, max_images=1000):
        """Download images for the dataset"""
        print(f"Downloading up to {max_images} images...")
        
        # Take a subset for faster prototyping
        df_subset = df.head(max_images).copy()
        
        successful_downloads = []
        failed_downloads = []
        
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Downloading images"):
            product_id = row['product_id']
            image_url = row['feature_image_s3']
            
            filepath = self.download_image(image_url, product_id)
            
            if filepath:
                successful_downloads.append({
                    'product_id': product_id,
                    'image_path': filepath,
                    'image_url': image_url
                })
            else:
                failed_downloads.append({
                    'product_id': product_id,
                    'image_url': image_url
                })
        
        print(f"Successfully downloaded: {len(successful_downloads)} images")
        print(f"Failed downloads: {len(failed_downloads)} images")
        
        # Filter dataframe to only include successfully downloaded images
        successful_product_ids = [item['product_id'] for item in successful_downloads]
        df_with_images = df_subset[df_subset['product_id'].isin(successful_product_ids)].copy()
        
        return df_with_images, successful_downloads, failed_downloads
    
    def save_processed_data(self, df, filename="products.json"):
        """Save processed data as JSON"""
        filepath = os.path.join(self.processed_dir, filename)
        
        # Convert to records format for JSON serialization
        records = df.to_dict('records')
        
        # Clean up any problematic data types
        for record in records:
            for key, value in record.items():
                # Handle pandas NA values
                if pd.isna(value) if not isinstance(value, (list, np.ndarray)) else False:
                    record[key] = None
                # Handle numpy integers
                elif isinstance(value, np.integer):
                    record[key] = int(value)
                # Handle numpy floats
                elif isinstance(value, np.floating):
                    record[key] = float(value)
                # Handle numpy arrays and lists
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                # Handle lists that might contain numpy types
                elif isinstance(value, list):
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, np.integer):
                            cleaned_list.append(int(item))
                        elif isinstance(item, np.floating):
                            cleaned_list.append(float(item))
                        else:
                            cleaned_list.append(item)
                    record[key] = cleaned_list
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"Processed data saved to {filepath}")
        return filepath

def main():
    # Initialize processor
    processor = DataProcessor()
    
    # Load datasets
    csv_files = ['jeans_bd_processed_data.csv', 'dresses_bd_processed_data.csv']
    df = processor.load_datasets(csv_files)
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Download images (start with 500 for quick prototyping)
    df_with_images, successful_downloads, failed_downloads = processor.download_images(df_clean, max_images=500)
    
    # Save processed data
    processor.save_processed_data(df_with_images)
    
    # Save download log
    download_log = {
        'successful': successful_downloads,
        'failed': failed_downloads,
        'total_processed': len(df_with_images)
    }
    
    with open(os.path.join(processor.processed_dir, 'download_log.json'), 'w') as f:
        json.dump(download_log, f, indent=2)
    
    print("Data processing complete!")
    print(f"Final dataset size: {len(df_with_images)} products")

if __name__ == "__main__":
    main() 