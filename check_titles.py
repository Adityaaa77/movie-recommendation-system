import pandas as pd
import os

base_path = r"C:\Aditya\5th Sem\Elevate Labs AIML"
data = pd.read_csv(base_path + r"\cleaned_movies.csv")

print("🎬 Total Movies:", data['title'].nunique())
print("\n📜 Some sample movie titles:\n")
print(data['title'].dropna().unique()[:20])
