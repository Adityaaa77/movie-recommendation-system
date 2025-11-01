# p2.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Step 1: Load cleaned dataset -----------
base_path = r"C:\Aditya\5th Sem\Elevate Labs AIML"
cleaned_path = base_path + r"\cleaned_movies.csv"

data = pd.read_csv(cleaned_path)

print("✅ Data loaded successfully:", data.shape)

# ----------- Step 2: Create User-Movie Rating Matrix -----------
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

print("🎞️ User-Movie matrix shape:", user_movie_matrix.shape)

# ----------- Step 3: Fill NaN with 0 (no rating = 0) -----------
user_movie_matrix = user_movie_matrix.fillna(0)

# ----------- Step 4: Compute similarity between users -----------
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=user_movie_matrix.index, 
                                  columns=user_movie_matrix.index)

print("🤝 User similarity matrix created!")

# ----------- Step 5: Recommendation Function -----------

def recommend_movies_for_user(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return f"User {user_id} not found!"

    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    
    # Movies watched by similar users
    similar_user_ids = similar_users.index
    similar_users_ratings = user_movie_matrix.loc[similar_user_ids]
    
    # Weighted average rating
    weighted_ratings = similar_users_ratings.T.dot(similar_users) / similar_users.sum()
    
    # Movies user has already seen
    user_seen = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    
    # Recommend unseen movies
    recommendations = weighted_ratings.drop(user_seen).sort_values(ascending=False).head(top_n)
    
    return recommendations

# ----------- Step 6: Test with a sample user -----------
sample_user = 1
recommended_movies = recommend_movies_for_user(sample_user, top_n=5)

print(f"\n🎯 Top 5 Movie Recommendations for User {sample_user}:")
print(recommended_movies)
