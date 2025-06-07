import json
import os
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import random

class OutfitRecommender:
    def __init__(self, data_dir="data"):
        """Initialize the outfit recommendation engine"""
        self.data_dir = data_dir
        
        # Load data
        self.products_data = None
        self.embeddings = None
        self.product_ids = None
        
        # Define compatibility rules
        self.compatibility_rules = {
            # Category ID mappings (based on the dataset)
            56: {  # Jeans/Bottoms
                'category_name': 'jeans',
                'compatible_with': ['tops', 'dresses', 'shoes', 'accessories'],
                'style_keywords': ['skinny', 'straight', 'wide', 'high-rise', 'mid-rise', 'low-rise']
            },
            # Add more categories as we discover them from the data
        }
        
        # Style compatibility matrix
        self.style_compatibility = {
            'casual': ['casual', 'smart-casual', 'relaxed'],
            'formal': ['formal', 'business', 'elegant'],
            'sporty': ['sporty', 'athletic', 'active'],
            'bohemian': ['bohemian', 'boho', 'free-spirited'],
            'minimalist': ['minimalist', 'clean', 'simple']
        }
        
        self._load_data()
        self._analyze_categories()
        
        print("Outfit recommender initialized successfully!")
    
    def _load_data(self):
        """Load product data and embeddings"""
        processed_dir = os.path.join(self.data_dir, "processed")
        embeddings_dir = os.path.join(self.data_dir, "embeddings")
        
        # Load product data
        products_file = os.path.join(processed_dir, "products_with_embeddings.json")
        with open(products_file, 'r', encoding='utf-8') as f:
            products_list = json.load(f)
        
        self.products_data = {product['product_id']: product for product in products_list}
        
        # Load embeddings and product IDs
        self.embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
        with open(os.path.join(embeddings_dir, "product_ids.json"), 'r') as f:
            self.product_ids = json.load(f)
        
        print(f"Loaded data for {len(self.products_data)} products")
    
    def _analyze_categories(self):
        """Analyze the dataset to understand category distribution"""
        category_counts = {}
        category_examples = {}
        
        for product in self.products_data.values():
            cat_id = product.get('category_id')
            if cat_id:
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
                if cat_id not in category_examples:
                    category_examples[cat_id] = []
                if len(category_examples[cat_id]) < 3:
                    category_examples[cat_id].append(product.get('product_name_clean', ''))
        
        print("Category analysis:")
        for cat_id, count in sorted(category_counts.items()):
            examples = ", ".join(category_examples.get(cat_id, []))
            print(f"  Category {cat_id}: {count} products (e.g., {examples})")
    
    def _extract_style_keywords(self, product):
        """Extract style keywords from product data"""
        keywords = set()
        
        # Extract from product name
        name = product.get('product_name_clean', '').lower()
        description = product.get('description_clean', '').lower()
        features = product.get('features_parsed', [])
        
        # Common style keywords
        style_terms = [
            'skinny', 'straight', 'wide', 'flare', 'bootcut',
            'high-rise', 'mid-rise', 'low-rise',
            'casual', 'formal', 'business', 'sporty',
            'vintage', 'modern', 'classic', 'trendy',
            'minimalist', 'bohemian', 'elegant'
        ]
        
        text_to_search = f"{name} {description} {' '.join(features)}"
        
        for term in style_terms:
            if term in text_to_search:
                keywords.add(term)
        
        return list(keywords)
    
    def _calculate_style_compatibility(self, product1, product2):
        """Calculate style compatibility between two products"""
        keywords1 = set(self._extract_style_keywords(product1))
        keywords2 = set(self._extract_style_keywords(product2))
        
        # Calculate overlap
        if not keywords1 or not keywords2:
            return 0.5  # Neutral compatibility if no keywords
        
        overlap = len(keywords1.intersection(keywords2))
        total = len(keywords1.union(keywords2))
        
        return overlap / total if total > 0 else 0.5
    
    def _get_visual_similarity(self, product_id1, product_id2):
        """Get visual similarity between two products"""
        try:
            idx1 = self.product_ids.index(product_id1)
            idx2 = self.product_ids.index(product_id2)
            
            emb1 = self.embeddings[idx1].reshape(1, -1)
            emb2 = self.embeddings[idx2].reshape(1, -1)
            
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
        except (ValueError, IndexError):
            return 0.0
    
    def _filter_by_category_compatibility(self, anchor_product, candidate_products):
        """Filter products based on category compatibility"""
        anchor_category = anchor_product.get('category_id')
        
        # For now, since we mainly have jeans, we'll use simple rules
        # In a full implementation, this would be more sophisticated
        
        compatible_products = []
        
        for product in candidate_products:
            candidate_category = product.get('category_id')
            
            # Don't recommend the same product
            if product['product_id'] == anchor_product['product_id']:
                continue
            
            # For jeans (category 56), recommend other jeans with different styles
            if anchor_category == 56 and candidate_category == 56:
                # Check if they have different style characteristics
                anchor_name = anchor_product.get('product_name_clean', '').lower()
                candidate_name = product.get('product_name_clean', '').lower()
                
                # Simple style differentiation
                if ('skinny' in anchor_name and 'wide' in candidate_name) or \
                   ('wide' in anchor_name and 'skinny' in candidate_name) or \
                   ('high-rise' in anchor_name and 'low-rise' in candidate_name):
                    compatible_products.append(product)
                elif anchor_name != candidate_name:  # Different products
                    compatible_products.append(product)
            
            # Add more category rules here as needed
        
        return compatible_products
    
    def recommend_outfit(self, product_id, num_recommendations=5) -> List[Dict]:
        """Recommend outfit items for a given product"""
        if product_id not in self.products_data:
            print(f"Product {product_id} not found")
            return []
        
        anchor_product = self.products_data[product_id]
        
        # Get all other products
        all_products = list(self.products_data.values())
        
        # Filter by category compatibility
        compatible_products = self._filter_by_category_compatibility(anchor_product, all_products)
        
        if not compatible_products:
            # Fallback: recommend visually similar products
            compatible_products = [p for p in all_products if p['product_id'] != product_id]
        
        # Score each compatible product
        scored_products = []
        
        for product in compatible_products:
            # Calculate different compatibility scores
            visual_sim = self._get_visual_similarity(product_id, product['product_id'])
            style_compat = self._calculate_style_compatibility(anchor_product, product)
            
            # Price compatibility (prefer similar price range)
            anchor_price = anchor_product.get('selling_price_clean', 0)
            candidate_price = product.get('selling_price_clean', 0)
            
            if anchor_price > 0 and candidate_price > 0:
                price_ratio = min(anchor_price, candidate_price) / max(anchor_price, candidate_price)
                price_compat = price_ratio
            else:
                price_compat = 0.5
            
            # Brand compatibility (same brand gets bonus)
            brand_compat = 1.0 if anchor_product.get('brand_clean') == product.get('brand_clean') else 0.7
            
            # Combined score
            total_score = (
                visual_sim * 0.4 +
                style_compat * 0.3 +
                price_compat * 0.2 +
                brand_compat * 0.1
            )
            
            scored_products.append({
                'product': product,
                'total_score': total_score,
                'visual_similarity': visual_sim,
                'style_compatibility': style_compat,
                'price_compatibility': price_compat,
                'brand_compatibility': brand_compat
            })
        
        # Sort by total score
        scored_products.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Prepare recommendations
        recommendations = []
        for i, item in enumerate(scored_products[:num_recommendations]):
            product = item['product']
            
            recommendation = {
                'rank': i + 1,
                'product_id': product['product_id'],
                'product_name': product.get('product_name_clean', ''),
                'brand': product.get('brand_clean', ''),
                'selling_price': product.get('selling_price_clean', 0),
                'image_url': product.get('feature_image_s3', ''),
                'compatibility_score': round(item['total_score'], 3),
                'visual_similarity': round(item['visual_similarity'], 3),
                'style_compatibility': round(item['style_compatibility'], 3),
                'price_compatibility': round(item['price_compatibility'], 3),
                'reason': self._generate_recommendation_reason(anchor_product, product, item)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_recommendation_reason(self, anchor_product, recommended_product, scores):
        """Generate a human-readable reason for the recommendation"""
        reasons = []
        
        if scores['visual_similarity'] > 0.8:
            reasons.append("visually similar style")
        
        if scores['style_compatibility'] > 0.6:
            reasons.append("compatible style elements")
        
        if scores['brand_compatibility'] > 0.9:
            reasons.append("same brand")
        
        if scores['price_compatibility'] > 0.7:
            reasons.append("similar price range")
        
        if not reasons:
            reasons.append("complementary piece")
        
        return "Recommended for " + " and ".join(reasons)
    
    def get_outfit_suggestions(self, product_ids: List[str]) -> Dict:
        """Get outfit suggestions for multiple products"""
        if not product_ids:
            return {}
        
        # For now, use the first product as anchor
        anchor_id = product_ids[0]
        recommendations = self.recommend_outfit(anchor_id)
        
        return {
            'anchor_product': self.products_data.get(anchor_id),
            'recommendations': recommendations,
            'outfit_theme': self._determine_outfit_theme(product_ids)
        }
    
    def _determine_outfit_theme(self, product_ids: List[str]) -> str:
        """Determine the overall theme of an outfit"""
        all_keywords = set()
        
        for product_id in product_ids:
            if product_id in self.products_data:
                keywords = self._extract_style_keywords(self.products_data[product_id])
                all_keywords.update(keywords)
        
        # Simple theme detection
        if 'formal' in all_keywords or 'business' in all_keywords:
            return "Professional"
        elif 'casual' in all_keywords:
            return "Casual"
        elif 'sporty' in all_keywords:
            return "Athletic"
        else:
            return "Everyday"

def test_recommender():
    """Test the outfit recommender"""
    print("Testing Outfit Recommender...")
    
    # Initialize recommender
    recommender = OutfitRecommender()
    
    # Get a random product to test with
    random_product_id = random.choice(list(recommender.products_data.keys()))
    anchor_product = recommender.products_data[random_product_id]
    
    print(f"\nTesting recommendations for:")
    print(f"Product: {anchor_product.get('product_name_clean')}")
    print(f"Brand: {anchor_product.get('brand_clean')}")
    print(f"Price: ${anchor_product.get('selling_price_clean', 0)}")
    
    # Get recommendations
    recommendations = recommender.recommend_outfit(random_product_id, num_recommendations=5)
    
    print(f"\nTop {len(recommendations)} outfit recommendations:")
    for rec in recommendations:
        print(f"{rec['rank']}. {rec['product_name']} by {rec['brand']}")
        print(f"   Score: {rec['compatibility_score']} | Price: ${rec['selling_price']}")
        print(f"   Reason: {rec['reason']}")
        print()
    
    print("Recommender test completed!")

if __name__ == "__main__":
    test_recommender() 