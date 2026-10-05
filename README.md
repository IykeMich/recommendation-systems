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

## Deploying (e.g. Vercel)

* **Frontend:** set the project's root directory to `<project>/frontend`, and set
  `NEXT_PUBLIC_API_URL` (`NEXT_PUBLIC_API_BASE_URL` for Project 1) to the deployed API.
* **Backend:** set the root directory to `<project>/backend`. It holds `api/`, `src/`,
  `data/` and `artifacts/`, so everything the API needs is included. Start command:
  `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. Serverless file systems are read-only
  and reset between requests. Read-only endpoints work, but anything that writes files
  won't persist there: feedback logs, new interactions, retraining, and the capstone's
  SQLite decision store. Use a database or object storage (e.g. DynamoDB or S3, as in
  each project's AWS design) for those.
