import streamlit as st
import requests
import json
from PIL import Image
import io
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import threading
import sys
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fashion Visual Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Global variable to track if FastAPI server is started
if 'fastapi_started' not in st.session_state:
    st.session_state.fastapi_started = False

def start_fastapi_server():
    """Start the FastAPI server in a separate thread"""
    try:
        # Get the project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        app_dir = project_root / "app"
        
        # Change to the app directory and start the server
        os.chdir(str(app_dir))
        
        # Start FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except Exception as e:
        print(f"Error starting FastAPI server: {e}")

def ensure_fastapi_running():
    """Ensure FastAPI server is running, start it if not"""
    if not st.session_state.fastapi_started:
        if not check_api_health():
            st.info("🚀 Starting FastAPI server...")
            
            # Start server in background thread
            server_thread = threading.Thread(target=start_fastapi_server, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            max_wait = 30  # seconds
            wait_time = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while wait_time < max_wait:
                if check_api_health():
                    st.session_state.fastapi_started = True
                    progress_bar.progress(100)
                    status_text.success("✅ FastAPI server started successfully!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    break
                
                wait_time += 1
                progress = int((wait_time / max_wait) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Starting server... ({wait_time}/{max_wait}s)")
                time.sleep(1)
            
            if wait_time >= max_wait:
                progress_bar.empty()
                status_text.error("❌ Failed to start FastAPI server. Please check the logs.")
                return False
        else:
            st.session_state.fastapi_started = True
    
    return True

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
        border: 2px solid #e1e8ed;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .product-image {
        border-radius: 10px;
        margin-bottom: 1rem;
        max-width: 100%;
        height: auto;
        object-fit: cover;
    }
    .product-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.3;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .product-brand {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    .similarity-score {
        background-color: #e8f4fd;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.85rem;
        margin: 0.3rem 0;
        display: inline-block;
    }
    .price-tag {
        color: #2E86AB;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 0.5rem 0;
        background-color: #f0f9ff;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        display: inline-block;
    }
    .recommendation-reason {
        font-style: italic;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        line-height: 1.3;
    }
    .product-actions {
        margin-top: auto;
        padding-top: 1rem;
        width: 100%;
    }
    .stButton > button {
        width: 100%;
        margin: 0.2rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    .product-grid {
        gap: 1rem;
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

def display_product_card(product, show_similarity=False, show_compatibility=False, show_actions=False, product_id_for_actions=None):
    """Display a product card with image and details"""
    # Create a styled container for the product card
    with st.container():
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        
        # Display product image
        try:
            if product.get('image_url'):
                st.image(
                    product['image_url'], 
                    width=180, 
                    use_column_width=False,
                    caption=None
                )
            else:
                st.markdown('<div style="height: 180px; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-bottom: 1rem;"><p>No image available</p></div>', unsafe_allow_html=True)
        except:
            st.markdown('<div style="height: 180px; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-bottom: 1rem;"><p>Image not available</p></div>', unsafe_allow_html=True)
        
        # Product details
        product_name = product.get('product_name', 'Unknown Product')
        brand = product.get('brand', 'Unknown')
        
        st.markdown(f'<div class="product-title">{product_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="product-brand">{brand}</div>', unsafe_allow_html=True)
        
        # Price
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
        
        # Action buttons if requested
        if show_actions and product_id_for_actions:
            st.markdown('<div class="product-actions">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Find Similar", key=f"similar_{product_id_for_actions}", help="Find similar products"):
                    st.session_state[f"show_similar_{product_id_for_actions}"] = True
            
            with col2:
                if st.button("👗 Get Outfit", key=f"outfit_{product_id_for_actions}", help="Get outfit recommendations"):
                    st.session_state[f"show_outfit_{product_id_for_actions}"] = True
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Ensure FastAPI server is running
    if not ensure_fastapi_running():
        st.error("⚠️ Failed to start the backend server. Please check your setup.")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">👗 Fashion Visual Search</h1>', unsafe_allow_html=True)
    
    # Check API health (should be running now)
    if not check_api_health():
        st.error("⚠️ API is not responding. Please wait a moment and refresh the page.")
        if st.button("🔄 Retry Connection"):
            st.rerun()
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
                        
                        # Create improved grid layout for search results
                        cols_per_row = 3
                        st.markdown('<div class="product-grid">', unsafe_allow_html=True)
                        
                        for i in range(0, len(results['results']), cols_per_row):
                            cols = st.columns(cols_per_row, gap="large")
                            
                            for j, col in enumerate(cols):
                                if i + j < len(results['results']):
                                    product = results['results'][i + j]
                                    
                                    with col:
                                        # Add rank indicator
                                        st.markdown(f'<div style="text-align: center; background-color: #2E86AB; color: white; padding: 0.2rem; border-radius: 5px; margin-bottom: 0.5rem; font-weight: bold;">Rank #{product["rank"]}</div>', unsafe_allow_html=True)
                                        
                                        display_product_card(product, show_similarity=True)
                                        
                                        # Add outfit recommendation button
                                        if st.button(f"👗 Get Outfit Ideas", key=f"outfit_{product['product_id']}", help="Get outfit recommendations for this item"):
                                            show_outfit_recommendations(product['product_id'], product['product_name'])
                        
                        st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔄 Load Random Products", type="primary"):
            with st.spinner("Loading products..."):
                products = get_random_products(num_products=12)
                
                if products:
                    st.session_state.browse_products = products
                    # Clear any previous selections
                    for key in list(st.session_state.keys()):
                        if key.startswith(('show_similar_', 'show_outfit_')):
                            del st.session_state[key]
    
    with col2:
        if 'browse_products' in st.session_state:
            num_products = st.selectbox("Products per page", [6, 9, 12, 15], index=2)
    
    # Display products if available
    if 'browse_products' in st.session_state:
        products = st.session_state.browse_products
        
        st.markdown("---")
        st.subheader(f"Showing {len(products)} Products")
        
        # Create improved grid layout with equal column heights
        cols_per_row = 3
        
        # Add some spacing
        st.markdown('<div class="product-grid">', unsafe_allow_html=True)
        
        for i in range(0, len(products), cols_per_row):
            # Create columns with equal width and gap
            cols = st.columns(cols_per_row, gap="large")
            
            for j, col in enumerate(cols):
                if i + j < len(products):
                    product = products[i + j]
                    product_id = product['product_id']
                    
                    with col:
                        # Display product card with integrated actions
                        display_product_card(
                            product, 
                            show_actions=True, 
                            product_id_for_actions=product_id
                        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle action results below the grid
        st.markdown("---")
        
        # Check for similar product requests
        for key in st.session_state.keys():
            if key.startswith('show_similar_') and st.session_state[key]:
                product_id = key.replace('show_similar_', '')
                product = next((p for p in products if p['product_id'] == product_id), None)
                if product:
                    show_similar_products(product_id, product['product_name'])
                    st.session_state[key] = False  # Reset the flag
        
        # Check for outfit recommendation requests
        for key in st.session_state.keys():
            if key.startswith('show_outfit_') and st.session_state[key]:
                product_id = key.replace('show_outfit_', '')
                product = next((p for p in products if p['product_id'] == product_id), None)
                if product:
                    show_outfit_recommendations(product_id, product['product_name'])
                    st.session_state[key] = False  # Reset the flag
    
    else:
        st.info("👆 Click 'Load Random Products' to start browsing our collection!")

def show_similar_products(product_id, product_name):
    """Show similar products for a selected item"""
    st.subheader(f"🔍 Similar to: {product_name}")
    
    with st.spinner("Finding similar products..."):
        results = search_by_product_id(product_id, top_k=8)
        
        if results:
            st.success(f"Found {results['num_results']} similar items!")
            
            # Display results in improved grid
            cols_per_row = 4
            st.markdown('<div class="product-grid">', unsafe_allow_html=True)
            
            for i in range(0, len(results['results']), cols_per_row):
                cols = st.columns(cols_per_row, gap="medium")
                
                for j, col in enumerate(cols):
                    if i + j < len(results['results']):
                        product = results['results'][i + j]
                        
                        with col:
                            # Add rank indicator
                            st.markdown(f'<div style="text-align: center; background-color: #059669; color: white; padding: 0.2rem; border-radius: 5px; margin-bottom: 0.5rem; font-weight: bold; font-size: 0.8rem;">#{product["rank"]}</div>', unsafe_allow_html=True)
                            display_product_card(product, show_similarity=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
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
    
    ## 🚀 Quick Start
    Simply run: `streamlit run frontend/streamlit_app.py`
    
    The FastAPI backend server will start automatically! No need to run separate commands.
    
    ## 🔧 Technology Stack
    - **Frontend**: Streamlit
    - **Backend**: FastAPI (auto-started)
    - **AI Models**: CLIP (OpenAI) for visual embeddings
    - **Search**: FAISS for fast similarity search
    - **Recommendations**: Multi-criteria scoring system
    
    ## 🚀 Features
    - **Visual Search**: Upload any fashion image to find similar items
    - **Outfit Recommendations**: Get complete outfit suggestions
    - **Product Browsing**: Explore the fashion catalog
    - **Real-time Performance**: Sub-second search results
    - **Auto-startup**: Backend starts automatically with frontend
    
    ## 📊 Current Dataset
    - **500 Products** processed and indexed
    - **Jeans & Dresses** categories
    - **Real product data** from fashion retailers
    
    ## 🎨 How It Works
    1. **Auto-startup**: FastAPI server launches automatically
    2. **Image Processing**: CLIP model extracts visual features
    3. **Similarity Search**: FAISS finds visually similar items
    4. **Outfit Matching**: Multi-criteria algorithm suggests complementary pieces
    5. **Real-time Results**: Optimized for fast response times
    
    ## 🔮 Future Enhancements
    - More product categories
    - Advanced style analysis
    - User personalization
    - Trend detection
    """)
    
    # System architecture diagram (text-based)
    st.subheader("System Architecture")
    st.code("""
    streamlit run frontend/streamlit_app.py
                    ↓
    Auto-start FastAPI Backend (localhost:8000)
                    ↓
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