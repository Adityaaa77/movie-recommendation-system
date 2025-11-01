# p3.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ----------- Step 1: Load cleaned dataset -----------
base_path = r"C:\Aditya\5th Sem\Elevate Labs AIML"
cleaned_path = base_path + r"\cleaned_movies.csv"

data = pd.read_csv(cleaned_path)
print("✅ Data loaded:", data.shape)

# ----------- Step 2: Clean up genres column -----------
def clean_genres(x):
    try:
        # Some genres are JSON-like strings
        genres_list = ast.literal_eval(x)
        if isinstance(genres_list, list):
            return " ".join([d['name'] for d in genres_list if 'name' in d])
    except:
        return x  # already string
    return x

data['genres'] = data['genres'].fillna('').apply(clean_genres)

# ----------- Step 3: Convert genres text to vectors -----------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['genres'])

print("📊 TF-IDF Matrix Shape:", tfidf_matrix.shape)

# ----------- Step 4: Compute similarity between movies -----------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ----------- Step 5: Mapping title → index -----------
indices = pd.Series(data.index, index=data['title'].str.lower()).drop_duplicates()

# ----------- Step 6: Recommendation function -----------
def recommend_by_genre(title, top_n=5):
    title = title.lower()
    if title not in indices:
        close_matches = [t for t in indices.index if title in t]
        if close_matches:
            title = close_matches[0]
        else:
            return [f"❌ Movie '{title}' not found in dataset!"]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices].tolist()

# ----------- Step 7: Test with sample movie -----------
sample_movie = "Toy Story"
recommendations = recommend_by_genre(sample_movie, top_n=5)

print(f"\n🎯 Because you liked '{sample_movie}', you may also like:")
for i, rec in enumerate(recommendations, start=1):
    print(f"{i}. {rec}")
