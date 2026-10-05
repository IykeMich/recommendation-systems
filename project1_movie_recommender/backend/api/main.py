"""RecLab local recommendation API (FastAPI).

Serves the Next.js playground UI with two toy recommenders built from the CSVs in backend/data/:
- Project 1 (movies): content-based recommendations via TF-IDF + cosine similarity, both for
  catalog movies and for a new, user-described movie (projected into the same TF-IDF space).
- Project 2 (retail): an item-item collaborative baseline with a popularity cold-start fallback.
Everything is loaded and precomputed in memory at import time; nothing is persisted.
"""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Backend root (one level up from backend/api/), so data paths work from any cwd. All CSVs live
# inside backend/data/, so deploying or bundling the backend folder includes them.
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
movies = pd.read_csv(DATA_DIR / "movies.csv")
# Retail data (a copy of Project 2's dataset) powers the playground's Retail tab.
products = pd.read_csv(DATA_DIR / "retail" / "products.csv")
events = pd.read_csv(DATA_DIR / "retail" / "interactions.csv")
users = pd.read_csv(DATA_DIR / "retail" / "users.csv")

# Project 1: actual TF-IDF content model.
# Each movie becomes one "document" of genres + description; TF-IDF weights distinctive words,
# and cosine_similarity yields a dense N x N movie-to-movie similarity matrix (rows = movies.index).
movie_text = movies["genres"].fillna("") + " " + movies["description"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english")
movie_matrix = vectorizer.fit_transform(movie_text)
movie_similarity = cosine_similarity(movie_matrix)

app = FastAPI(title="RecLab Recommendation API")

# Allow the local Next.js frontend to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://127.0.0.1:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class EventIn(BaseModel):
    """Request body for POST /v1/events: one user-item interaction (e.g. a view or add-to-cart).

    `event_value` is the interaction weight (the UI sends 1 for view, 3 for cart).
    """
    user_id: str
    item_id: str
    event_type: str
    event_value: float = 1.0
    project: str = "retail"

class MovieQueryIn(BaseModel):
    """Request body for POST /v1/recommendations/content/query: a new movie that is not in the catalog.

    `genres` is pipe-separated like movies.csv (e.g. "Drama|Comedy"); `title` is display-only.
    """
    title: str = ""
    genres: str = ""
    description: str = ""

@app.get("/health")
def health():
    """Liveness check; the UI polls this to show the "API online/offline" badge."""
    return {"status": "ok"}

@app.get("/v1/movies")
def search_movies(search: str = "", limit: int = 12):
    """Case-insensitive catalog search; returns the total match count plus the first `limit` movies."""
    # Catalog lookup for the playground UI: match on title, genres or item_id.
    matches = movies
    if search.strip():
        term = search.strip().lower()
        # Substring match on title/genres (regex=False so input like "(" is literal); ID is a prefix match.
        matches = movies[
            movies.title.str.lower().str.contains(term, regex=False)
            | movies.genres.str.lower().str.contains(term, regex=False)
            | movies.item_id.str.lower().str.startswith(term)
        ]
    columns = ["item_id", "title", "genres", "description", "year", "language"]
    return {
        "total": int(len(matches)),
        "movies": matches.head(limit)[columns].to_dict("records"),
    }

@app.get("/v1/retail/users/{user_id}")
def retail_user_profile(user_id: str):
    """Return a retail shopper's profile row plus how many interaction events they have (404 if unknown)."""
    profile = users[users.user_id == user_id]
    if profile.empty:
        raise HTTPException(404, "Unknown user_id")
    return {
        **profile.iloc[0].to_dict(),
        "event_count": int((events.user_id == user_id).sum()),
    }

@app.get("/v1/recommendations/content/{item_id}")
def content_recs(item_id: str, limit: int = 10):
    """More-like-this: the `limit` movies most similar to `item_id` by TF-IDF cosine similarity."""
    if item_id not in set(movies.item_id):
        raise HTTPException(404, "Unknown item_id")

    # Row of the similarity matrix for this movie, sorted highest-similarity first.
    idx = movies.index[movies.item_id == item_id][0]
    ranked = movie_similarity[idx].argsort()[::-1]
    # Drop the source movie itself (it is always ~1.0 similar to itself), then keep the top `limit`.
    ranked = [i for i in ranked if movies.iloc[i].item_id != item_id][:limit]

    return {
        "strategy": "content_based_tfidf",
        "source_item_id": item_id,
        "recommendations": [
            {
                "item_id": movies.iloc[i].item_id,
                "title": movies.iloc[i].title,
                "score": round(float(movie_similarity[idx, i]), 4),
                "genres": movies.iloc[i].genres,
            }
            for i in ranked
        ],
    }

@app.post("/v1/recommendations/content/query")
def content_query_recs(query: MovieQueryIn, limit: int = 10):
    """Cold-start item: the `limit` catalog movies most similar to a movie described in the request."""
    if not (query.genres.strip() or query.description.strip()):
        raise HTTPException(400, "Provide genres or a description")

    # Same text recipe as the catalog, transformed with the already-fitted vectorizer (no refit),
    # so the new movie lands in the same TF-IDF space as every catalog movie.
    query_vector = vectorizer.transform([query.genres + " " + query.description])
    # Words the vectorizer never saw are dropped; with none left every similarity would be 0.
    if query_vector.nnz == 0:
        raise HTTPException(400, "None of these words appear in the catalog vocabulary")

    scores = cosine_similarity(query_vector, movie_matrix)[0]
    # Highest similarity first, skipping movies that share no words with the query.
    ranked = [i for i in scores.argsort()[::-1] if scores[i] > 0][:limit]

    return {
        "strategy": "content_based_tfidf_query",
        "source": {"title": query.title, "genres": query.genres},
        "recommendations": [
            {
                "item_id": movies.iloc[i].item_id,
                "title": movies.iloc[i].title,
                "score": round(float(scores[i]), 4),
                "genres": movies.iloc[i].genres,
            }
            for i in ranked
        ],
    }

@app.get("/v1/recommendations/user/{user_id}")
def user_recs(user_id: str, limit: int = 10):
    """Personalised retail recommendations for `user_id`.

    Users with interaction history get item-item collaborative scores; anyone else gets popular items.
    """
    # Cold start: users with no events get the products with the highest total event_value instead.
    if user_id not in set(events.user_id):
        pop = (
            events.groupby("item_id")["event_value"]
            .sum()
            .sort_values(ascending=False)
            .head(limit)
            .index.tolist()
        )
        # Note: isin() keeps the products.csv row order, not the popularity order computed above.
        recs = products[products.item_id.isin(pop)][
            ["item_id", "name", "category"]
        ].to_dict("records")
        return {
            "strategy": "popular_cold_start",
            "user_id": user_id,
            "recommendations": recs,
        }

    # Items this user already interacted with are excluded from the recommendations.
    seen = set(events.loc[events.user_id == user_id, "item_id"])
    user_events = events[events.user_id == user_id]

    # Transparent item-item collaborative baseline.
    # For each item the user touched, find "peers" who touched the same item, then credit every other
    # item those peers touched with (user's event weight x peer's event weight). Items co-interacted
    # with by many overlapping shoppers accumulate the highest scores. Simple but O(n^2) via iterrows.
    scores = {}
    for _, e in user_events.iterrows():
        peers = set(events.loc[events.item_id == e.item_id, "user_id"])
        for _, n in events[events.user_id.isin(peers)].iterrows():
            if n.item_id in seen:
                continue
            scores[n.item_id] = scores.get(n.item_id, 0.0) + (
                float(e.event_value) * float(n.event_value)
            )

    # Highest scores first; join product names/categories, skipping item_ids missing from the catalog.
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    product_map = products.set_index("item_id")
    recs = [
        {
            "item_id": item_id,
            "name": product_map.loc[item_id, "name"],
            "category": product_map.loc[item_id, "category"],
            "score": round(float(score), 4),
        }
        for item_id, score in ranked
        if item_id in product_map.index
    ]
    return {
        "strategy": "collaborative_item_item_baseline",
        "user_id": user_id,
        "recommendations": recs,
    }

@app.post("/v1/events")
def record_event(event: EventIn):
    """Ingest a user interaction event and echo it back (not stored, so recommendations don't change)."""
    # Local harness: accept the event.
    # AWS version sends it to the event pipeline/Kinesis.
    return {"accepted": True, "event": event.model_dump()}
