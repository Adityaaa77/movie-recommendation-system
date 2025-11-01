import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Title
st.title("🎬 Movie Recommendation System (Content-Based)")

# Load data
movies = pd.read_csv("movies.csv")
st.success("✅ Data loaded successfully!")

# TF-IDF setup
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
st.write(f"📊 TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# Similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie recommendation logic
def get_recommendations(title, cosine_sim=cosine_sim):
    indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()
    title = title.lower()
    if title not in indices:
        return ["❌ Movie not found in dataset!"]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# User input
movie_name = st.text_input("🎥 Enter a movie name:")
if movie_name:
    recs = get_recommendations(movie_name)
    st.subheader(f"🎯 Because you liked '{movie_name}', you may also like:")
    for i, r in enumerate(recs, 1):
        st.write(f"{i}. {r}")

# Footer
st.markdown("---")
st.markdown("💡 **Made by Aditya Rajpal**")
