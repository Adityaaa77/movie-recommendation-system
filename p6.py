import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🎥 Hybrid Movie Recommendation System")

# Load data
movies = pd.read_csv("movies.csv")
st.success("✅ Data loaded successfully!")

# TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Simple hybrid-style placeholder
def recommend(title):
    indices = pd.Series(movies.index, index=movies['title'].str.lower()).drop_duplicates()
    title = title.lower()
    if title not in indices:
        return ["❌ Movie not found in dataset!"]
    idx = indices[title]
    sim_scores = list(enumerate(content_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

movie_input = st.text_input("🎬 Enter a movie you liked:")
if movie_input:
    recs = recommend(movie_input)
    st.subheader(f"🎯 Recommended Movies similar to '{movie_input}':")
    for i, r in enumerate(recs, 1):
        st.write(f"{i}. {r}")

# Footer
st.markdown("---")
st.markdown("🌟 **Made by Aditya Rajpal**")
