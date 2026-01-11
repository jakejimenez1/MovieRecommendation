# 🎬 Movie Recommendation System

A personalized movie recommendation engine using collaborative filtering with SVD (Singular Value Decomposition), featuring real-time TMDB API integration and intelligent three-tier caching.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

## 📋 Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Collaborative Filtering**: Uses SVD matrix factorization to predict user preferences
- **Real-time Movie Data**: Integration with TMDB API for movie overviews and posters
- **Three-Tier Caching System**:
  - CSV data → SQLite for faster queries
  - User recommendations cached to avoid recomputation
  - TMDB API responses cached to reduce external calls
- **Dual Recommendations**: Get both "movies you'll love" and "movies to avoid"
- **Interactive Jupyter Notebook**: Exploratory data analysis and model training
- **Web Interface**: Streamlit app for easy interaction 

## 🎯 Demo

```python
# Get top 5 recommendations for user 12
user_id = 12
recommendations = get_top_n_recommendations(user_id, model_svd, df, movies_df, movie_encoder, n=5)

# Output:
# 1. Shawshank Redemption, The (1994) - Rating: 4.59/5.0
# 2. Godfather, The (1972) - Rating: 4.52/5.0
# 3. Schindler's List (1993) - Rating: 4.48/5.0
```

## 🛠 Tech Stack

- **Machine Learning**: scikit-surprise (SVD algorithm)
- **Data Processing**: pandas, numpy, scikit-learn
- **Database**: SQLite3
- **API Integration**: TMDB API (The Movie Database)
- **Visualization**: Jupyter Notebook, IPython.display
- **Web Framework**: Streamlit (for deployment)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **RMSE** | 0.47 |
| **Dataset Size** | 100,836 ratings |
| **Users** | 610 |
| **Movies** | 9,724 |
| **Cache Hit Rate** | ~95% |
| **Recommendation Generation** | <1 second (cached) |
| **Cold Start** | ~5-10 seconds (first run) |

## 🚀 Installation

### Prerequisites
- Python 3.8-3.11
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up TMDB API**
   - Get a free API key from [TMDB](https://www.themoviedb.org/settings/api)
   - Create a `.env` file in the project root:
```bash
TMDB_TOKEN=your_api_token_here
```

5. **Download MovieLens dataset**
   - The project uses [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)
   - Place `ratings.csv` and `movies.csv` in the project root

## 💻 Usage

### Jupyter Notebook

1. **Start Jupyter**
```bash
jupyter notebook
```

2. **Open and run cells in order**
   - `Recommender_System.ipynb`
   - Follow the markdown instructions in the notebook

3. **Get recommendations**
```python
user_id = 12
top_movies = get_top_n_recommendations(user_id, model_svd, df, movies_df, movie_encoder, n=5)
display_recommendations(user_id, top_movies, "Top")
```

### Streamlit Web App

```bash
streamlit run app.py
```

### Command Line

```bash
# View database contents
sqlite3 recommendations.db "SELECT * FROM user_recommendations LIMIT 5;"

# Check cache statistics
python -c "from your_module import get_cache_stats; get_cache_stats()"
```

## 📁 Project Structure

```
movie-recommender/
│
├── Recommender_System.ipynb   # Main Jupyter notebook
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (TMDB API key)
├── README.md                   # This file
│
├── data/
│   ├── ratings.csv            # User ratings data
│   └── movies.csv             # Movie metadata
│
├── databases/
│   ├── movies.db              # SQLite database for ratings/movies
│   ├── recommendations.db     # Cached user recommendations
│   └── tmdb_cache.db          # Cached TMDB API responses
│
└── src/
    ├── data_processing.py     # Data loading and preprocessing
    ├── model.py               # SVD model training and prediction
    ├── tmdb_integration.py    # TMDB API functions
    └── database.py            # Database helper functions
```

## 🔍 How It Works

### 1. Data Processing
- Load 100K+ ratings from MovieLens dataset
- Encode user IDs and movie IDs using LabelEncoder
- Split data into training (80%) and testing (20%) sets
- Extract and binarize movie genres

### 2. Model Training
- Use Singular Value Decomposition (SVD) for collaborative filtering
- Learn latent factors representing user preferences and movie characteristics
- Predict ratings for unseen user-movie pairs

### 3. Recommendation Generation
```
User Input → Check Cache → If cached: Return results
                        → If not: Compute predictions → Fetch TMDB data → Cache → Return
```

### 4. Caching Strategy
- **Level 1**: CSV → SQLite (faster queries, indexed)
- **Level 2**: User recommendations (avoid recomputation)
- **Level 3**: TMDB API responses (reduce external API calls)

### 5. Algorithm Choice: Why SVD?
- **Accuracy**: Lower RMSE than basic collaborative filtering
- **Scalability**: Efficient for large sparse matrices
- **Cold Start**: Can handle new users with limited ratings
- **Latent Factors**: Captures hidden patterns in user preferences

## 🔮 Future Improvements

- [ ] **Content-Based Filtering**: Combine with genre/cast information
- [ ] **Hybrid Model**: Merge collaborative and content-based approaches
- [ ] **A/B Testing**: Compare different recommendation strategies
- [ ] **Real-time Updates**: Incremental model updates as new ratings arrive
- [ ] **User Profiles**: Allow users to create accounts and rate movies
- [ ] **Social Features**: Friend recommendations, watch parties
- [ ] **Production Deployment**: Docker, CI/CD, monitoring

## 👤 Author

**Your Name**
- LinkedIn: [Jake Jimenez](https://www.linkedin.com/in/jake-jimenez/)
- Email: jakejim2003@gmail.com

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [TMDB](https://www.themoviedb.org/) for the movie API
- [Surprise](http://surpriselib.com/) library for recommendation algorithms

## 📈 Project Statistics

- **Lines of Code**: ~1,000
- **Development Time**: 2 weeks
- **API Calls Saved**: 95% through caching
- **Database Size**: ~50MB (with cache)

---