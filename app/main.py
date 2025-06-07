from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
from typing import List, Optional
import uvicorn
import os
from pathlib import Path

from search_engine import VisualSearchEngine
from recommender import OutfitRecommender

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Visual Search API",
    description="AI-powered fashion search and outfit recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for engines (initialized on startup)
search_engine = None
outfit_recommender = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine and recommender on startup"""
    global search_engine, outfit_recommender
    
    print("Initializing Fashion Search API...")
    
    try:
        # Ensure we're in the correct directory
        from pathlib import Path
        
        # Get the directory where this script is located
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        data_dir = project_root / "data"
        
        print(f"Working directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
        print(f"Data directory: {data_dir}")
        
        # Initialize search engine with absolute data path
        print("Loading visual search engine...")
        search_engine = VisualSearchEngine(data_dir=str(data_dir))
        
        # Initialize outfit recommender with absolute data path
        print("Loading outfit recommender...")
        outfit_recommender = OutfitRecommender(data_dir=str(data_dir))
        
        print("Fashion Search API initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing API: {e}")
        import traceback
        traceback.print_exc()
        raise e

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fashion Visual Search API",
        "version": "1.0.0",
        "endpoints": {
            "visual_search": "/search/visual",
            "product_search": "/search/product/{product_id}",
            "outfit_recommendations": "/recommend/outfit/{product_id}",
            "product_info": "/product/{product_id}",
            "random_products": "/products/random",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "search_engine_loaded": search_engine is not None,
        "recommender_loaded": outfit_recommender is not None
    }

@app.post("/search/visual")
async def visual_search(
    file: UploadFile = File(...),
    top_k: int = 10
):
    """
    Search for similar products using an uploaded image
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    # Validate file type
    content_type = getattr(file, 'content_type', None)
    if not content_type or not content_type.startswith('image/'):
        # Additional check by filename extension
        filename = getattr(file, 'filename', '')
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Perform search
        start_time = time.time()
        results = search_engine.search_by_image(image, top_k=top_k)
        search_time = time.time() - start_time
        
        return {
            "success": True,
            "search_time": round(search_time, 3),
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/product/{product_id}")
async def search_by_product(
    product_id: str,
    top_k: int = 10
):
    """
    Search for products similar to a given product ID
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        results = search_engine.search_by_product_id(product_id, top_k=top_k)
        
        if not results:
            raise HTTPException(status_code=404, detail="Product not found or no similar products")
        
        return {
            "success": True,
            "product_id": product_id,
            "num_results": len(results),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/recommend/outfit/{product_id}")
async def get_outfit_recommendations(
    product_id: str,
    num_recommendations: int = 5
):
    """
    Get outfit recommendations for a given product
    """
    if not outfit_recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    try:
        recommendations = outfit_recommender.recommend_outfit(
            product_id, 
            num_recommendations=num_recommendations
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="Product not found or no recommendations available")
        
        # Get anchor product info
        anchor_product = outfit_recommender.products_data.get(product_id)
        
        return {
            "success": True,
            "anchor_product": {
                "product_id": product_id,
                "product_name": anchor_product.get('product_name_clean', '') if anchor_product else '',
                "brand": anchor_product.get('brand_clean', '') if anchor_product else '',
                "price": anchor_product.get('selling_price_clean', 0) if anchor_product else 0,
                "image_url": anchor_product.get('feature_image_s3', '') if anchor_product else ''
            },
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/product/{product_id}")
async def get_product_info(product_id: str):
    """
    Get detailed information for a specific product
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        product_info = search_engine.get_product_info(product_id)
        
        if not product_info:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return {
            "success": True,
            "product": product_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get product info: {str(e)}")

@app.get("/products/random")
async def get_random_products(num_products: int = 10):
    """
    Get random products for browsing/demo purposes
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        if num_products > 50:
            num_products = 50  # Limit to prevent abuse
        
        products = search_engine.get_random_products(num_products)
        
        return {
            "success": True,
            "num_products": len(products),
            "products": products
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get random products: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """
    Get system statistics and performance metrics
    """
    try:
        stats = {}
        
        if search_engine:
            stats["search_engine"] = search_engine.get_stats()
        
        if outfit_recommender:
            stats["recommender"] = {
                "total_products": len(outfit_recommender.products_data),
                "categories_analyzed": len(set(p.get('category_id') for p in outfit_recommender.products_data.values())),
                "embedding_dimension": outfit_recommender.embeddings.shape[1] if outfit_recommender.embeddings is not None else 0
            }
        
        stats["api_info"] = {
            "version": "1.0.0",
            "status": "operational",
            "uptime": time.time()
        }
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/search/batch")
async def batch_visual_search(
    files: List[UploadFile] = File(...),
    top_k: int = 5
):
    """
    Perform visual search on multiple images
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for i, file in enumerate(files):
        content_type = getattr(file, 'content_type', None)
        if not content_type or not content_type.startswith('image/'):
            # Additional check by filename extension
            filename = getattr(file, 'filename', '')
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue
        
        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Perform search
            search_results = search_engine.search_by_image(image, top_k=top_k)
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": True,
                "num_results": len(search_results),
                "results": search_results
            })
            
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "batch_size": len(files),
        "results": results
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 