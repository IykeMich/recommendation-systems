# Project 1 — RecLab Movie Recommender (content-based)

The first project in the lab pack. A FastAPI backend serves a **content-based
movie recommender** (TF-IDF over genres + description, ranked by cosine
similarity) to a Next.js + TanStack Query playground. In the Movies tab you can
pick a catalog movie, or switch to **Describe new** and type a title, genres and
description for a movie that isn't in the catalog: its text is projected into
the same TF-IDF space and matched against every catalog movie (the content-based
answer to item cold start; nothing is saved). The playground also has a
**Retail** tab: a simple item-item collaborative baseline over a copy of the
Project 2 retail data, as a preview of Project 2. A static **Project details**
page at `/about` (linked from the header) summarises the project for recruiters
and hiring managers and works even when the API is down.

```
movies.csv → genres + description text → TF-IDF vectors → cosine similarity
  → "movies like this one" → FastAPI → Next.js playground
```

## Structure

| Path | What it holds |
| --- | --- |
| `backend/api/main.py` | FastAPI app (`app`): content recommendations, movie search, retail recommendations, shopper profiles, events |
| `backend/data/movies.csv`, `backend/data/movie_interactions.csv` | Project 1 movie catalog and interactions (synthetic) |
| `backend/data/movies1.csv`, `backend/data/movies2.csv` | Alternative catalog files kept for reference; not read by the API or the training script |
| `backend/data/retail/` | `products.csv`, `interactions.csv`, `users.csv`: a copy of the current Project 2 retail data (598 products, 1,000 shoppers, 27,811 events), used by the Retail tab. Copy them again whenever Project 2's data is regenerated. The previous copy is kept in `archive/` |
| `backend/model_code/train_content_model.py` | Builds the TF-IDF artifact (`backend/model_code/artifacts/project1_tfidf.joblib`, gitignored) for deployment |
| `backend/aws/` | SAM template (`aws/sam/template.yaml`) and Lambda handler (`aws/lambda/recommendation_api/`) that forwards requests to a SageMaker endpoint |
| `backend/requirements.txt` | Backend dependencies |
| `frontend/` | Next.js + TanStack Query playground (Movies and Retail tabs) plus the `/about` Project details page |
| `docs/api-contract.md` | The stable API contract the frontend relies on |

Everything the backend reads lives inside `backend/`, so deploying or bundling
that folder never leaves data behind. For hosting, set the backend service's
root directory to `project1_movie_recommender/backend` and start it with
`uvicorn api.main:app --host 0.0.0.0 --port $PORT`; the frontend's root
directory is `project1_movie_recommender/frontend` (set `NEXT_PUBLIC_API_BASE_URL`
to the deployed API, and add the frontend's origin to `allow_origins` in
`backend/api/main.py`, which only allows `localhost:3001` today).

## Run it locally

```bash
# API (terminal 1)
cd project1_movie_recommender/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8001
# Swagger: http://localhost:8001/docs, e.g. GET /v1/recommendations/content/M00001
# or POST /v1/recommendations/content/query with {"genres": "Sci-Fi|Thriller", "description": "astronauts stranded on a distant planet"}

# Frontend (terminal 2)
cd project1_movie_recommender/frontend
npm install
cp .env.example .env.local      # NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
npm run dev                     # http://localhost:3001 (Project details: /about)
```

Optional: build the model artifact from `backend/` with
`python model_code/train_content_model.py` (using the backend's virtual
environment); it writes `model_code/artifacts/project1_tfidf.joblib`.

Each project has its own ports (Project 1: 8001/3001, Project 2: 8002/3002, Project 3: 8003/3003,
Project 4: 8004/3004, capstone: 8005/3005), so they can all run at the same time.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/v1/movies?search=` | Search the movie catalog by title, genre or ID |
| GET | `/v1/recommendations/content/{item_id}?limit=10` | Movies most similar to `item_id` (TF-IDF cosine) |
| POST | `/v1/recommendations/content/query?limit=10` | Movies most similar to a new movie described in the body (`title`, `genres`, `description`); 400 if genres and description are both empty or share no words with the catalog |
| GET | `/v1/recommendations/user/{user_id}?limit=10` | Retail recommendations (item-item baseline; popularity for unknown users) |
| GET | `/v1/retail/users/{user_id}` | Retail shopper profile plus event count |
| POST | `/v1/events` | Accepts an event (echoed back locally; the AWS version sends it to the event pipeline) |

## AWS path

The AWS version keeps the same application contract but moves inference behind
API Gateway + Lambda + SageMaker: `backend/aws/sam/template.yaml` deploys an HTTP API
and `backend/aws/lambda/recommendation_api/handler.py`, which forwards the request body
to the SageMaker endpoint named in `SAGEMAKER_ENDPOINT`. The Lambda forwards the
body unchanged, so the new-movie query (`POST /v1/recommendations/content/query`)
needs no Lambda change, but the SageMaker inference code must accept a text
query and run it through the trained vectorizer the same way `backend/api/main.py`
does.

To deploy it (from `backend/`): train the artifact with
`python model_code/train_content_model.py`, host it on a SageMaker endpoint, then
`cd aws/sam && sam build && sam deploy --guided`, setting `SAGEMAKER_ENDPOINT` in
`template.yaml` (currently `replace-me`) to your endpoint name. The SAM
`CodeUri` (`../lambda/recommendation_api/`) is relative to `aws/sam/`, so run SAM
from there.
