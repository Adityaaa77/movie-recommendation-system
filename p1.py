# p1.py
import pandas as pd
import os

# ----------- Step 1: Path set karo -----------
base_path = r"C:\Aditya\5th Sem\Elevate Labs AIML"

movies_path = os.path.join(base_path, "movies.csv")
ratings_path = os.path.join(base_path, "ratings.csv")

# ----------- Step 2: Load datasets -----------
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

print("🎬 Movies shape:", movies.shape)
print("⭐ Ratings shape:", ratings.shape)

# ----------- Step 3: Rename column for merging -----------
movies.rename(columns={'id': 'movieId'}, inplace=True)

# ----------- Step 4: Merge both -----------
movie_data = pd.merge(ratings, movies, on="movieId", how="inner")

# ----------- Step 5: Clean & save -----------
movie_data.drop_duplicates(inplace=True)
movie_data.fillna("", inplace=True)

print("✅ Data merged successfully!")
print(movie_data.head(3))

# Save cleaned version
cleaned_path = os.path.join(base_path, "cleaned_movies.csv")
movie_data.to_csv(cleaned_path, index=False)
print(f"💾 Cleaned dataset saved at: {cleaned_path}")
