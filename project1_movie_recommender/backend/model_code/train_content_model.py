"""Train and save the Project 1 TF-IDF content model as a deployable artifact.

Run from backend/ (works from anywhere): python model_code/train_content_model.py
Writes backend/model_code/artifacts/project1_tfidf.joblib (e.g. to package for SageMaker).
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Backend root (one level up from model_code/); the movie catalog lives in backend/data/.
ROOT = Path(__file__).resolve().parents[1]
movies = pd.read_csv(ROOT / "data" / "movies.csv")
# Same text recipe as the API: genres + description form one document per movie.
text = movies["genres"].fillna("") + " " + movies["description"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(text)
similarity = cosine_similarity(matrix)

OUT = ROOT / "model_code" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# Save the fitted vectorizer and TF-IDF matrix with the item IDs in row order.
joblib.dump(
    {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "item_ids": movies["item_id"].tolist(),
    },
    OUT / "project1_tfidf.joblib",
)

print("Saved:", OUT / "project1_tfidf.joblib")
print("Items:", len(movies))
