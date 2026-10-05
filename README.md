# AWS Recommendation Systems Lab Pack

Five self-contained projects. Each folder has its own data, backend, frontend,
AWS files and README, so any one of them can be run, deployed or bundled on its own.

Every project follows the same layout:

```
<project>/
  README.md
  frontend/        Next.js app (includes a /about "Project details" page)
  backend/
    api/main.py    FastAPI app — run `uvicorn api.main:app` from backend/
    src/ tests/ data/ artifacts/ aws/ notebooks/ ...
    requirements.txt
```

Each app's header has a **Project details** badge that links to `/about`, a one-page
summary of the problem, approach, results and tech stack for recruiters and reviewers.

| Folder | Project | Backend entry point |
| --- | --- | --- |
| [`project1_movie_recommender/`](project1_movie_recommender/) | Content-based movie recommender (TF-IDF) and the RecLab playground | `backend/api/main.py` |
| [`project2_shopsmart/`](project2_shopsmart/) | ShopSmart: item-item collaborative filtering | `backend/api/main.py` |
| [`project3_context_offers/`](project3_context_offers/) | Context-aware offer recommender | `backend/api/main.py` |
| [`project4_financial_recommender/`](project4_financial_recommender/) | Eligibility-aware, explainable financial products | `backend/api/main.py` |
| [`capstone_fraud_detection/`](capstone_fraud_detection/) | Fraud detection and prevention platform | `backend/api/main.py` |

## Data

Each project keeps its CSVs in its own `backend/data/` folder. Nothing reads from outside its project.

| Project | Files |
| --- | --- |
| Project 1 | `movies.csv`, `movie_interactions.csv`, plus `retail/` (a copy of Project 2's data for the Retail tab) |
| Project 2 | `products.csv`, `users.csv`, `interactions.csv` |
| Project 3 | `offers.csv`, `offer_interactions.csv`, plus `users.csv` (Project 2's retail users; same IDs) |
| Project 4 | `financial_products.csv`, `users.csv`, `interactions.csv` |
| Capstone | `raw/transactions.csv` |

All data is synthetic and intentionally patterned for learning and evaluation. It is
safe to publish with a portfolio. Do not present the synthetic fraud label as a
real-world fraud model.

## Deploying

* **Frontend (e.g. Vercel):** set the project's root directory to `<project>/frontend`, and set
  `NEXT_PUBLIC_API_URL` (`NEXT_PUBLIC_API_BASE_URL` for Project 1) to the deployed API.
* **Backend (e.g. Render):** [`render.yaml`](render.yaml) is a Render Blueprint with one web
  service per backend (in Render: **New → Blueprint**, pick this repo). To set one up by hand
  instead, use:
  * Root directory: `<project>/backend`. It holds `api/`, `src/`, `data/` and `artifacts/`, so
    everything the API needs is included.
  * Build: `pip install -r requirements.txt`
  * Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  * Health check: `/health`
  * Environment: `PYTHON_VERSION=3.11.9`. Every API always allows `http://localhost` /
    `http://127.0.0.1` on any port and any `https://*.vercel.app` URL (production and preview
    deployments), so no CORS setting is needed for local testing or Vercel. For a custom frontend
    domain, also set `CORS_ORIGINS` (comma-separated). Locally the APIs still run on ports 8001–8005; on Render each service listens
    on Render's `$PORT` and the frontend calls its `https://<service>.onrender.com` URL.

  `requirements.txt` pins the library versions the committed `.joblib` models were trained
  with (scikit-learn 1.6.1, numpy 2.0.2). Keep the pins, or retrain after upgrading, otherwise the
  models may fail to load. numpy 2.0.2 has no Python 3.13 wheels, which is why Python is pinned to 3.11.
* **Written data resets.** Render's disk is temporary, so feedback logs, new interactions,
  retrained models and the capstone's SQLite decision store reset to the committed state on every
  deploy or restart. That's fine for a demo. For persistence, use a Render disk, a database or object
  storage (e.g. DynamoDB or S3, as in each project's AWS design).
* **Don't host the backends on Vercel.** Serverless file systems are read-only, so endpoints
  that write files fail (in ShopSmart that includes recommendations, which log an impression).
