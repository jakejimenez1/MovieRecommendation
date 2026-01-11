import streamlit as st
import pandas as pd
import sqlite3
import pickle
import os
from surprise import SVD
import requests
import re

## Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        color: #1f1f1f;
    }
    .rating {
        color: #f39c12;
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        background-color: #e50914;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

## TMDB API Setup
from dotenv import load_dotenv
load_dotenv()

TMDB_TOKEN = os.getenv("TMDB_TOKEN")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
HEADERS = {"Authorization": f"Bearer {TMDB_TOKEN}"}

## Helper functions
@st.cache_resource
## Load model and necessary data
def load_model_and_data():
    try:
        ## Check if model exists, otherwise train it
        if os.path.exists('model_svd.pkl'):
            with open('model_svd.pkl', 'rb') as f:
                model_svd = pickle.load(f)
        else:
            st.warning("Model not found. Please train the model first in the Jupyter notebook.")
            return None, None, None, None, None
        
        ## Load encoders
        if os.path.exists('encoders.pkl'):
            with open('encoders.pkl', 'rb') as f:
                encoders = pickle.load(f)
                user_encoder = encoders['user_encoder']
                movie_encoder = encoders['movie_encoder']
        else:
            st.warning("Encoders not found. Please save encoders from notebook.")
            return None, None, None, None, None
        
        ## Load data from database
        conn = sqlite3.connect('movies.db')
        ratings_df = pd.read_sql('SELECT * FROM ratings', conn)
        movies_df = pd.read_sql('SELECT * FROM movies', conn)
        conn.close()
        
        ## Prepare df 
        df = pd.merge(ratings_df, movies_df[['movieId', 'genres']], on='movieId', how='left')
        df['userId'] = user_encoder.transform(df['userId'])
        df['movieId'] = movie_encoder.transform(df['movieId'])
        
        return model_svd, df, movies_df, user_encoder, movie_encoder
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None
## Remove year from title
def clean_movie_title(title):
    title = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
    return title.strip()

## Extract year from title
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else None

## Get TMDB info with caching
def get_tmdb_info_cached(title, movie_id):
    movie_id = int(movie_id)
    
    conn = None
    try:
        conn = sqlite3.connect('tmdb_cache.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT overview, poster_url FROM tmdb_cache WHERE movie_id = ?', (movie_id,))
        result = cursor.fetchone()
        
        if result:
            return result[0], result[1]
        
        ## Fetch from API
        year = extract_year(title)
        clean_title = clean_movie_title(title)
        
        params = {"query": clean_title, "include_adult": False}
        if year:
            params["year"] = year
        
        response = requests.get(f"{BASE_URL}/search/movie", headers=HEADERS, params=params, timeout=10)
        data = response.json()
        
        if data.get('results'):
            movie = data['results'][0]
            overview = movie.get('overview', 'No overview available')
            poster_path = movie.get('poster_path')
            poster_url = IMAGE_BASE + poster_path if poster_path else None
        else:
            overview = "No overview found"
            poster_url = None
        
        ## Cache it
        cursor.execute('''
            INSERT OR REPLACE INTO tmdb_cache (movie_id, title, overview, poster_url)
            VALUES (?, ?, ?, ?)
        ''', (movie_id, title, overview, poster_url))
        conn.commit()
        
        return overview, poster_url
    except Exception as e:
        return "Error fetching overview", None
    finally:
        if conn:
            conn.close()

## Get cached recommendations
def get_recommendations_cached(user_id, rec_type="top", n=5):
    conn = None
    try:
        conn = sqlite3.connect('recommendations.db')
        query = '''
            SELECT movie_id as movieId, title, predicted_rating, overview, poster_url
            FROM user_recommendations 
            WHERE user_id = ? AND recommendation_type = ?
            ORDER BY predicted_rating DESC 
            LIMIT ?
        '''
        cursor = conn.cursor()
        cursor.execute(query, (user_id, rec_type, n))
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results if results else None
    finally:
        if conn:
            conn.close()

## Generate top recommendations
def get_top_n_recommendations(user_id, model_svd, df, movies_df, movie_encoder, n=5):
    ## Check cache first
    cached = get_recommendations_cached(user_id, "top", n)
    if cached:
        return cached
    
    ## Compute recommendations
    user_movies = df[df['userId'] == user_id]['movieId'].unique()
    all_movies = df['movieId'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))
    
    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)
    
    top_n_recommendations = sorted(predictions_cf, key=lambda x: x.est, reverse=True)[:n]
    
    results = []
    for pred in top_n_recommendations:
        movie_id_encoded = int(pred.iid)
        predicted_rating = pred.est
        
        movie_id_original = movie_encoder.inverse_transform([movie_id_encoded])[0]
        movie_row = movies_df[movies_df['movieId'] == movie_id_original]
        
        if movie_row.empty:
            continue
        
        title = movie_row['title'].values[0]
        overview, poster_url = get_tmdb_info_cached(title, movie_id_original)
        
        results.append({
            "movieId": movie_id_original,
            "title": title,
            "predicted_rating": predicted_rating,
            "overview": overview,
            "poster_url": poster_url
        })
    
    ## Save to cache
    save_to_cache(user_id, results, "top")
    
    return results

# Generate "movies to avoid" recommendations
def get_bottom_n_recommendations(user_id, model_svd, df, movies_df, movie_encoder, n=5):
    # Check cache first
    cached = get_recommendations_cached(user_id, "bottom", n)
    if cached:
        return cached
    
    # Compute recommendations
    user_movies = df[df['userId'] == user_id]['movieId'].unique()
    all_movies = df['movieId'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))
    
    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)
    
    # Get BOTTOM predictions (LOWEST ratings - No reverse = true)
    bottom_n_recommendations = sorted(predictions_cf, key=lambda x: x.est)[:n]
    
    results = []
    for pred in bottom_n_recommendations:
        movie_id_encoded = int(pred.iid)
        predicted_rating = pred.est
        
        movie_id_original = movie_encoder.inverse_transform([movie_id_encoded])[0]
        movie_row = movies_df[movies_df['movieId'] == movie_id_original]
        
        if movie_row.empty:
            continue
        
        title = movie_row['title'].values[0]
        overview, poster_url = get_tmdb_info_cached(title, movie_id_original)
        
        results.append({
            "movieId": movie_id_original,
            "title": title,
            "predicted_rating": predicted_rating,
            "overview": overview,
            "poster_url": poster_url
        })
    
    # Save to cache
    save_to_cache(user_id, results, "bottom")
    
    return results

## Save recommendations to cache
def save_to_cache(user_id, recommendations, rec_type):
    conn = None
    try:
        conn = sqlite3.connect('recommendations.db')
        cursor = conn.cursor()
        
        for rec in recommendations:
            cursor.execute('''
                INSERT OR REPLACE INTO user_recommendations 
                (user_id, movie_id, predicted_rating, title, overview, poster_url, recommendation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, rec['movieId'], rec['predicted_rating'], 
                  rec['title'], rec['overview'], rec['poster_url'], rec_type))
        
        conn.commit()
    finally:
        if conn:
            conn.close()

## Main App
def main():
    ## Header
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Discover your next favorite movie using AI-powered recommendations")
    
    ## Load model and data
    with st.spinner("Loading model and data..."):
        model_svd, df, movies_df, user_encoder, movie_encoder = load_model_and_data()
    
    if model_svd is None:
        st.error("⚠️ Unable to load model. Please train the model first using the Jupyter notebook.")
        st.stop()
    
    ## Sidebar
    st.sidebar.header("⚙️ Settings")
    
    ## Get available users
    available_users = sorted(user_encoder.classes_.tolist())
    
    ## User selection
    user_id_input = st.sidebar.selectbox(
        "Select User ID",
        options=available_users,
        index=0
    )
    
    ## Encode user ID
    user_id_encoded = user_encoder.transform([user_id_input])[0]
    
    ## Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        step=1
    )
    
    ## Recommendation type
    rec_type = st.sidebar.radio(
        "Recommendation Type",
        options=["Top Picks for You", "Movies to Avoid"],
        index=0
    )
    
    ## Force refresh
    force_refresh = st.sidebar.checkbox("Force refresh (bypass cache)", value=False)
    
    ## Stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Stats")
    
    try:
        conn = sqlite3.connect('tmdb_cache.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM tmdb_cache')
        cached_movies = cursor.fetchone()[0]
        conn.close()
        st.sidebar.metric("Cached Movies", cached_movies)
    except:
        st.sidebar.metric("Cached Movies", "N/A")
    
    try:
        conn = sqlite3.connect('recommendations.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM user_recommendations')
        cached_users = cursor.fetchone()[0]
        conn.close()
        st.sidebar.metric("Users with Recommendations", cached_users)
    except:
        st.sidebar.metric("Users with Recommendations", "N/A")
    
    ## Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Recommendations for User {user_id_input}")
    
    with col2:
        if st.button("🔍 Get Recommendations", use_container_width=True):
            st.session_state.generate = True
    
    ## Generate recommendations
    if 'generate' not in st.session_state:
        st.session_state.generate = False
    
    if st.session_state.generate or force_refresh:
        with st.spinner("Generating recommendations..."):
            try:
                if force_refresh:
                    ## Clear cache for this user
                    conn = sqlite3.connect('recommendations.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM user_recommendations WHERE user_id = ?', (user_id_encoded,))
                    conn.commit()
                    conn.close()
                # Determine which type of recommendations to show
                if rec_type == "Top Picks for You":
                    recommendations = get_top_n_recommendations(
                        user_id_encoded, 
                        model_svd, 
                        df, 
                        movies_df, 
                        movie_encoder, 
                        n=n_recommendations
                    )
                    display_type = "Top Picks"
                else:
                    recommendations = get_bottom_n_recommendations(
                        user_id_encoded, 
                        model_svd, 
                        df, 
                        movies_df, 
                        movie_encoder, 
                        n=n_recommendations
                    )
                    display_type = "Movies to Avoid"
                
                if not recommendations:
                    st.warning("No recommendations found for this user.")
                    return
                
                ## Display recommendations
                st.markdown("---")
                
                for idx, movie in enumerate(recommendations, 1):
                    col_img, col_info = st.columns([1, 3])

                    # Displays the movie poster if movie has one
                    with col_img:
                        if movie['poster_url']:
                            st.image(movie['poster_url'], use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                    
                    st.markdown(f"### {idx}. {movie['title']}")
                        
                    # Show rating with color coding
                    rating = movie['predicted_rating']
                    if display_type == "Movies to Avoid":
                        # Low ratings in red for movies to avoid
                        rating_color = "#d00" if rating < 2.5 else "#f39c12"
                        rating_emoji = "👎" if rating < 2.5 else "⚠️"
                    else:
                        # High ratings in green for recommendations
                        rating_color = "#2ecc71" if rating >= 4.0 else "#f39c12"
                        rating_emoji = "⭐" if rating >= 4.0 else "✨"
                    
                    st.markdown(
                        f"<span style='color: {rating_color}; font-size: 18px;'>{rating_emoji} Predicted Rating: {rating:.2f}/5.0</span>", 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Overview:** {movie['overview']}")
                    
                    # Genre/year info
                    year = extract_year(movie['title'])
                    if year:
                        st.caption(f"📅 Released: {year}")
                    
                    st.markdown("---")
                
                st.success(f"✅ Generated {len(recommendations)} recommendations!")
                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
    
    else:
        ## Welcome message
        st.info("👆 Click 'Get Recommendations' to see personalized movie suggestions!")
        
        ## Show some stats
        st.markdown("---")
        st.subheader("📈 How it works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1️⃣ Collaborative Filtering**")
            st.write("Uses SVD to learn patterns from 100K+ ratings")
        
        with col2:
            st.markdown("**2️⃣ Real-time Data**")
            st.write("Fetches movie info from TMDB API")
        
        with col3:
            st.markdown("**3️⃣ Smart Caching**")
            st.write("95% faster through intelligent caching")

if __name__ == "__main__":
    main()