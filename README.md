# 👗 Fashion Visual Search & Outfit Recommendation System

An AI-powered fashion search and recommendatiFSon system using Streamlit + FastAPI + OpenCV + CLIP + FAISS + GNN + Local Filesystem.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-orange.svg)

## 🎯 Overview

This system enables users to:

- **Upload any fashion image** and find visually similar items
- **Get intelligent outfit recommendations** based on style compatibility
- **Browse product catalogs** with real-time search capabilities
- **Analyze system performance** with built-in monitoring

## 🚀 Features

### Core Functionality

- ✅ **Visual Search**: Upload images to find similar fashion items
- ✅ **Outfit Recommendations**: Multi-criteria scoring for complete outfits
- ✅ **Product Browsing**: Interactive catalog exploration
- ✅ **Real-time Performance**: Sub-second search results
- ✅ **System Analytics**: Performance monitoring and statistics

### Technical Highlights

- **CLIP Model Integration**: OpenAI's CLIP for visual embeddings
- **FAISS Vector Search**: Fast similarity search with 500+ products
- **Multi-criteria Recommendations**: Visual similarity + style compatibility + price matching
- **RESTful API**: Complete FastAPI backend with 8 endpoints
- **Interactive Frontend**: Beautiful Streamlit interface

## 🏗️ Architecture

```
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
```

## 📊 Dataset

- **500 Products** processed and indexed
- **Categories**: Jeans and Dresses
- **Real Data**: Sourced from fashion retailers
- **Features**: Product names, brands, prices, descriptions, images

## 🛠️ Technology Stack

### Backend

- **FastAPI**: High-performance web framework
- **CLIP**: OpenAI's vision-language model
- **FAISS**: Facebook's similarity search library
- **scikit-learn**: Machine learning utilities
- **Pandas**: Data processing

### Frontend

- **Streamlit**: Interactive web applications
- **Plotly**: Data visualization
- **PIL**: Image processing

### AI/ML

- **Transformers**: Hugging Face model library
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing

## 📋 Prerequisites

- Python 3.8+
- pip

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/Aadik1ng/Visual_Search.git
cd Visual_Search
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run frontend/streamlit_app.py --server.headless true --server.port 8501
```

That's it! The FastAPI backend will start automatically. No need to run separate commands.

The application will be available at:

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000 (auto-started)

## 📱 Features

### 🔍 Visual Search

- Upload any fashion image to find similar items
- CLIP-powered visual similarity matching
- Sub-second search results

### 🛍️ Outfit Recommendations

- Get complete outfit suggestions for any item
- Multi-criteria compatibility scoring
- Style, price, and brand matching

### 📊 Product Browsing

- Explore the fashion catalog
- Random product discovery
- Detailed product information

### 📈 System Monitoring

- Real-time performance metrics
- System health monitoring
- API statistics

## 📁 Project Structure

```
Visual_Search/
├── app/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── search_engine.py   # CLIP + FAISS search
│   └── recommender.py     # Outfit recommendations
├── frontend/              # Streamlit frontend
│   └── streamlit_app.py   # Main UI application
├── scripts/               # Data processing scripts
│   ├── data_processor.py  # CSV to JSON processing
│   └── feature_extractor.py # CLIP embeddings
├── data/                  # Data storage
│   ├── processed/         # Processed product data
│   ├── embeddings/        # CLIP embeddings
│   └── images/           # Product images
└── requirements.txt       # Python dependencies
```

## 🔌 API Endpoints

The FastAPI backend provides the following endpoints:

### Core Search

- `POST /search/visual` - Visual search by image upload
- `GET /search/product/{product_id}` - Find similar products
- `POST /search/batch` - Batch visual search

### Recommendations

- `GET /recommend/outfit/{product_id}` - Get outfit recommendations

### Product Data

- `GET /product/{product_id}` - Get product details
- `GET /products/random` - Get random products

### System

- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /` - API documentation

## 🛠️ Development

### Manual Backend Start (Optional)

If you need to run the backend separately:

```bash
cd app
python main.py
```

### Data Processing (Optional)

To reprocess the dataset:

```bash
python scripts/data_processor.py
python scripts/feature_extractor.py
```

## 🔍 Troubleshooting

### Common Issues

1. **"Module not found" errors**

   - Ensure all dependencies are installed: `pip install -r requirements.txt`
2. **"No such file or directory" errors**

   - Make sure you're running from the project root directory
3. **Backend startup fails**

   - Check if port 8000 is available
   - Verify all data files exist in `data/` directory
4. **Slow initial startup**

   - First run downloads CLIP model (~500MB)
   - Subsequent runs are much faster

### Performance Tips

- **First Run**: Allow 2-3 minutes for model download and initialization
- **Memory**: System uses ~2GB RAM for optimal performance
- **Storage**: Requires ~1GB for models and data

## 🚀 Deployment

### Local Development

```bash
streamlit run frontend/streamlit_app.py
```

### Production Deployment

For production deployment, consider:

- Using Docker containers
- Setting up reverse proxy (nginx)
- Configuring proper CORS origins
- Adding authentication if needed

## 📈 Future Enhancements

- [ ] More product categories (shoes, accessories, etc.)
- [ ] Advanced style analysis and trend detection
- [ ] User personalization and preferences
- [ ] Social features and outfit sharing
- [ ] Mobile app development
- [ ] Real-time inventory integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Facebook AI for FAISS
- Streamlit and FastAPI communities
- Fashion dataset providers

---

**Built with ❤️ for fashion enthusiasts and AI developers**

## 📞 Support

For support and questions:

- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation at `/docs`

---

**Built with ❤️ for the fashion industry**
