import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# 1️⃣ Load Data
# ----------------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("✅ Data loaded successfully!")

# ----------------------------
# 2️⃣ Merge both datasets
# ----------------------------
merged_df = pd.merge(ratings, movies, left_on='movieId', right_on='id')
print(f"🎬 Total combined records: {merged_df.shape}")

# ----------------------------
# 3️⃣ Create user-item matrix for collaborative filtering
# ----------------------------
user_movie_matrix = merged_df.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# User similarity
user_similarity = cosine_similarity(user_movie_matrix)
print("🤝 User similarity matrix ready!")

# ----------------------------
# 4️⃣ Content-based using TF-IDF
# ----------------------------
movies['overview'] = movies['overview'].fillna('')
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
tfidf_matrix = tfidf.fit_transform(movies['overview'])
content_similarity = cosine_similarity(tfidf_matrix)
print("🧩 Content similarity matrix ready!")

# ----------------------------
# 5️⃣ Hybrid Recommendation Function
# ----------------------------
def hybrid_recommendations(movie_title, user_id):
    # check if movie exists
    if movie_title not in movies['title'].values:
        print(f"❌ Movie '{movie_title}' not found in dataset!")
        return
    
    # find index of the movie
    idx = movies[movies['title'] == movie_title].index[0]

    # content similarity scores
    content_scores = list(enumerate(content_similarity[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    # collaborative (user-based) recommendations
    user_sim_scores = list(enumerate(user_similarity[user_id - 1]))
    user_sim_scores = sorted(user_sim_scores, key=lambda x: x[1], reverse=True)

    # combine both (weighted sum)
    hybrid_scores = {}
    for movie_idx, sim in content_scores[:50]:
        hybrid_scores[movie_idx] = hybrid_scores.get(movie_idx, 0) + 0.6 * sim  # 60% content weight

    for i, sim in user_sim_scores[:50]:
        user_ratings = user_movie_matrix.iloc[i]
        for j, rating in enumerate(user_ratings):
            if rating > 0:
                hybrid_scores[j] = hybrid_scores.get(j, 0) + 0.4 * sim  # 40% user weight

    # sort by score
    sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n🎯 Hybrid Recommendations for User {user_id} based on '{movie_title}':")
    count = 0
    for movie_idx, _ in sorted_recs:
        title = user_movie_matrix.columns[movie_idx]
        if title != movie_title:
            print(f"{count+1}. {title}")
            count += 1
        if count == 5:
            break


# ----------------------------
# 6️⃣ Run example
# ----------------------------
hybrid_recommendations("Mission: Impossible", user_id=1)
