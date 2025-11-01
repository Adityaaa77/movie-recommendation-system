import pandas as pd
import os

base_path = r"C:\Aditya\5th Sem\Elevate Labs AIML"

movies_path = os.path.join(base_path, "movies.csv")
ratings_path = os.path.join(base_path, "ratings.csv")

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

print("🎬 Movies columns:", movies.columns.tolist())
print("⭐ Ratings columns:", ratings.columns.tolist())
