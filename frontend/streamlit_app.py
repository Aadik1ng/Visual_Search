import streamlit as st
import requests
import json
from PIL import Image
import io
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fashion Visual Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        background-color: #f9f9f9;
    }
    .similarity-score {
        background-color: #e8f4fd;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .price-tag {
        color: #2E86AB;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .recommendation-reason {
        font-style: italic;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_random_products(num_products=12):
    """Get random products from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/products/random?num_products={num_products}")
        if response.status_code == 200:
            return response.json()["products"]
        return []
    except:
        return []

def search_by_image(image_file, top_k=10):
    """Perform visual search using uploaded image"""
    try:
        files = {"file": image_file}
        params = {"top_k": top_k}
        response = requests.post(f"{API_BASE_URL}/search/visual", files=files, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None

def search_by_product_id(product_id, top_k=10):
    """Search for similar products by product ID"""
    try:
        response = requests.get(f"{API_BASE_URL}/search/product/{product_id}?top_k={top_k}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_outfit_recommendations(product_id, num_recommendations=5):
    """Get outfit recommendations for a product"""
    try:
        response = requests.get(f"{API_BASE_URL}/recommend/outfit/{product_id}?num_recommendations={num_recommendations}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_system_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            return response.json()["stats"]
        return None
    except:
        return None

def display_product_card(product, show_similarity=False, show_compatibility=False):
    """Display a product card with image and details"""
    # Use container instead of columns to avoid nesting issues
    with st.container():
        # Display product image
        try:
            if product.get('image_url'):
                st.image(product['image_url'], width=150, caption=product.get('product_name', 'Product'))
            else:
                st.write("No image available")
        except:
            st.write("Image not available")
        
        # Product details
        st.markdown(f"**{product.get('product_name', 'Unknown Product')}**")
        st.markdown(f"*Brand:* {product.get('brand', 'Unknown')}")
        
        price = product.get('selling_price', 0)
        if price > 0:
            st.markdown(f'<div class="price-tag">${price:.2f}</div>', unsafe_allow_html=True)
        
        # Show similarity score if available
        if show_similarity and 'similarity_score' in product:
            score = product['similarity_score']
            st.markdown(f'<div class="similarity-score">Similarity: {score:.3f}</div>', unsafe_allow_html=True)
        
        # Show compatibility score if available
        if show_compatibility and 'compatibility_score' in product:
            score = product['compatibility_score']
            st.markdown(f'<div class="similarity-score">Compatibility: {score:.3f}</div>', unsafe_allow_html=True)
            
            if 'reason' in product:
                st.markdown(f'<div class="recommendation-reason">{product["reason"]}</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">👗 Fashion Visual Search</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the FastAPI server first.")
        st.code("python app/main.py")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Visual Search", "🛍️ Browse Products", "📊 System Stats", "ℹ️ About"]
    )
    
    if page == "🔍 Visual Search":
        visual_search_page()
    elif page == "🛍️ Browse Products":
        browse_products_page()
    elif page == "📊 System Stats":
        system_stats_page()
    elif page == "ℹ️ About":
        about_page()

def visual_search_page():
    st.header("Visual Search")
    st.write("Upload an image to find similar fashion items and get outfit recommendations!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a fashion image to search for similar items"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", width=300)
        
        with col2:
            st.subheader("Search Settings")
            top_k = st.slider("Number of results", min_value=5, max_value=20, value=10)
            
            if st.button("🔍 Search Similar Items", type="primary"):
                with st.spinner("Searching for similar items..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Perform search
                    results = search_by_image(uploaded_file, top_k=top_k)
                    
                    if results:
                        st.success(f"Found {results['num_results']} similar items in {results['search_time']} seconds!")
                        
                        # Display results
                        st.subheader("Search Results")
                        
                        # Create columns for grid layout
                        cols_per_row = 3
                        for i in range(0, len(results['results']), cols_per_row):
                            cols = st.columns(cols_per_row)
                            
                            for j, col in enumerate(cols):
                                if i + j < len(results['results']):
                                    product = results['results'][i + j]
                                    
                                    with col:
                                        st.markdown(f"**Rank {product['rank']}**")
                                        display_product_card(product, show_similarity=True)
                                        
                                        # Add outfit recommendation button
                                        if st.button(f"Get Outfit Ideas", key=f"outfit_{product['product_id']}"):
                                            show_outfit_recommendations(product['product_id'], product['product_name'])

def show_outfit_recommendations(product_id, product_name):
    """Show outfit recommendations for a selected product"""
    st.subheader(f"Outfit Recommendations for: {product_name}")
    
    with st.spinner("Getting outfit recommendations..."):
        recommendations = get_outfit_recommendations(product_id, num_recommendations=5)
        
        if recommendations:
            # Show anchor product
            st.write("**Selected Item:**")
            anchor = recommendations['anchor_product']
            display_product_card(anchor)
            
            st.write("**Complete the Look:**")
            
            # Display recommendations
            for rec in recommendations['recommendations']:
                with st.expander(f"Recommendation {rec['rank']}: {rec['product_name']}"):
                    display_product_card(rec, show_compatibility=True)
        else:
            st.warning("No outfit recommendations available for this item.")

def browse_products_page():
    st.header("Browse Products")
    st.write("Explore our fashion collection and find similar items!")
    
    # Get random products
    if st.button("🔄 Load Random Products"):
        with st.spinner("Loading products..."):
            products = get_random_products(num_products=12)
            
            if products:
                st.session_state.browse_products = products
    
    # Display products if available
    if 'browse_products' in st.session_state:
        products = st.session_state.browse_products
        
        st.subheader(f"Showing {len(products)} Products")
        
        # Create grid layout
        cols_per_row = 3
        for i in range(0, len(products), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(products):
                    product = products[i + j]
                    
                    with col:
                        display_product_card(product)
                        
                        # Buttons for actions (avoid nested columns)
                        if st.button("Find Similar", key=f"similar_{product['product_id']}"):
                            show_similar_products(product['product_id'], product['product_name'])
                        
                        if st.button("Get Outfit", key=f"browse_outfit_{product['product_id']}"):
                            show_outfit_recommendations(product['product_id'], product['product_name'])

def show_similar_products(product_id, product_name):
    """Show similar products for a selected item"""
    st.subheader(f"Similar to: {product_name}")
    
    with st.spinner("Finding similar products..."):
        results = search_by_product_id(product_id, top_k=8)
        
        if results:
            st.success(f"Found {results['num_results']} similar items!")
            
            # Display results in grid
            cols_per_row = 4
            for i in range(0, len(results['results']), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(results['results']):
                        product = results['results'][i + j]
                        
                        with col:
                            st.markdown(f"**#{product['rank']}**")
                            display_product_card(product, show_similarity=True)
        else:
            st.warning("No similar products found.")

def system_stats_page():
    st.header("System Statistics")
    st.write("Performance metrics and system information")
    
    # Get stats
    stats = get_system_stats()
    
    if stats:
        # API Info
        st.subheader("API Information")
        api_info = stats.get('api_info', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Version", api_info.get('version', 'Unknown'))
        with col2:
            st.metric("Status", api_info.get('status', 'Unknown'))
        with col3:
            uptime = api_info.get('uptime', 0)
            st.metric("Uptime", f"{uptime:.0f}s")
        
        # Search Engine Stats
        if 'search_engine' in stats:
            st.subheader("Search Engine")
            search_stats = stats['search_engine']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Products", search_stats.get('total_products', 0))
            with col2:
                st.metric("Embedding Dimension", search_stats.get('embedding_dimension', 0))
            with col3:
                st.metric("Model", search_stats.get('model_name', 'Unknown'))
            with col4:
                st.metric("Device", search_stats.get('device', 'Unknown'))
        
        # Recommender Stats
        if 'recommender' in stats:
            st.subheader("Recommendation Engine")
            rec_stats = stats['recommender']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Products", rec_stats.get('total_products', 0))
            with col2:
                st.metric("Categories", rec_stats.get('categories_analyzed', 0))
            with col3:
                st.metric("Embedding Dim", rec_stats.get('embedding_dimension', 0))
        
        # Performance visualization
        st.subheader("System Performance")
        
        # Create sample performance data (in a real system, this would come from logs)
        performance_data = {
            'Metric': ['Search Time', 'Recommendation Time', 'API Response'],
            'Average (ms)': [150, 80, 200],
            'Target (ms)': [500, 200, 300]
        }
        
        df = pd.DataFrame(performance_data)
        
        fig = px.bar(df, x='Metric', y=['Average (ms)', 'Target (ms)'], 
                     title="Performance Metrics", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Unable to fetch system statistics")

def about_page():
    st.header("About Fashion Visual Search")
    
    st.markdown("""
    ## 🎯 Overview
    This is an AI-powered fashion search and recommendation system that uses computer vision 
    and machine learning to help users find similar fashion items and create complete outfits.
    
    ## 🔧 Technology Stack
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **AI Models**: CLIP (OpenAI) for visual embeddings
    - **Search**: FAISS for fast similarity search
    - **Recommendations**: Multi-criteria scoring system
    
    ## 🚀 Features
    - **Visual Search**: Upload any fashion image to find similar items
    - **Outfit Recommendations**: Get complete outfit suggestions
    - **Product Browsing**: Explore the fashion catalog
    - **Real-time Performance**: Sub-second search results
    
    ## 📊 Current Dataset
    - **500 Products** processed and indexed
    - **Jeans & Dresses** categories
    - **Real product data** from fashion retailers
    
    ## 🎨 How It Works
    1. **Image Processing**: CLIP model extracts visual features
    2. **Similarity Search**: FAISS finds visually similar items
    3. **Outfit Matching**: Multi-criteria algorithm suggests complementary pieces
    4. **Real-time Results**: Optimized for fast response times
    
    ## 🔮 Future Enhancements
    - More product categories
    - Advanced style analysis
    - User personalization
    - Trend detection
    """)
    
    # System architecture diagram (text-based)
    st.subheader("System Architecture")
    st.code("""
    User Upload → Streamlit Frontend → FastAPI Backend
                                           ↓
    CLIP Model ← Image Processing ← Visual Search Engine
        ↓                              ↓
    Embeddings → FAISS Index → Similar Products
                                           ↓
    Outfit Recommender ← Multi-criteria Scoring
        ↓
    Recommendations → JSON Response → Frontend Display
    """)

if __name__ == "__main__":
    main() 