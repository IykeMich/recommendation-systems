# Project 4 — Financial Product Recommender

An eligibility-aware, explainable recommender. Hard rules decide which products
*may* be recommended; a model only ranks what is left; every displayed reason is
traceable to data. Built from the *Project 4 Financial Product Recommender Full
Build Guide*.

> **Learning project on synthetic data.** Not a credit-underwriting engine,
> financial advice or real bank eligibility policy.

```
user profile + history
  → HARD ELIGIBILITY RULES   "can this product enter the candidate set?"
  → eligible candidates
  → logistic-regression ranker   "how relevant is each one?"
  → Top-K → evidence-backed explanations → FastAPI → Next.js
  → feedback (request ID, strategy, model and rule versions) → retraining
```

## Structure

Everything Python lives in `backend/`; everything Next.js lives in `frontend/`.

| Path | What it holds |
| --- | --- |
| `backend/data/` | `users.csv`, `financial_products.csv`, `interactions.csv` (the Project 4 financial dataset) |
| `backend/notebooks/01_financial_recommender.ipynb` | Guided walkthrough calling the same `src/` code |
| `backend/src/config.py` | Paths, schema adapters (`USER_MAP`, `PRODUCT_MAP`, `INTERACTION_MAP`), label rule, Top-K |
| `backend/src/data.py` | Loading and schema mapping |
| `backend/src/eligibility.py` | **The rule layer**, versioned, testable with no model loaded |
| `backend/src/candidates.py` | Eligible candidates plus availability filtering |
| `backend/src/features.py` | Point-in-time user, product and history features; training examples |
| `backend/src/model.py` | The sklearn ranker and readable coefficients |
| `backend/src/explanations.py` | Deterministic reasons with evidence, and `verify_reason` |
| `backend/src/recommender.py` | `FinancialRecommender`: eligibility → rank → explain, plus cold start and what-if |
| `backend/src/evaluate.py` | Temporal evaluation: quality and policy metrics for 4 strategies |
| `backend/src/train.py` | Trains, evaluates and writes `backend/artifacts/` |
| `backend/src/experiments.py` | The guide's experiments (sections 33 and 44) |
| `backend/api/main.py` | FastAPI app |
| `backend/artifacts/` | Trained ranker, feature columns, metadata and `metrics.json` |
| `backend/requirements.txt`, `backend/requirements-dev.txt` | Runtime dependencies; dev adds pytest and httpx |
| `frontend/` | Next.js + TanStack Query app (`/` is the recommender, `/about` is the Project details page) |
| `backend/aws/sagemaker/`, `backend/aws/lambda/` | SageMaker train/inference; Lambda handlers for inference and Kinesis feedback |
| `backend/tests/` | pytest: eligibility (no model), recommender, API |

## Run it locally

Python commands run from `project4_financial_recommender/backend/`:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate    # once
pip install -r requirements-dev.txt                    # once

python -m src.train                                    # train + evaluate → artifacts/
uvicorn api.main:app --reload --port 8004              # terminal 1, Swagger at /docs

cd ../frontend && npm install && cp .env.example .env.local && npm run dev   # terminal 2
```

Tests: `python -m pytest -q`. Experiments: `python -m src.experiments` (both from `backend/`).
The notebook expects to be opened from `backend/notebooks/`.
Each project has its own ports (Project 1: 8001/3001, Project 2: 8002/3002, Project 3: 8003/3003,
Project 4: 8004/3004, capstone: 8005/3005), so they can all run at the same time.

## What the data says (inspect first)

* **The schema differs from the guide's examples.** Products have `min_income` and `risk_level`,
  not region, age limits or segment. Users have `monthly_income`, `employment`, `risk_profile`,
  `age_band`, `region` and `existing_products`. Events are `view`, `learn` and `apply`.
* **The rules come from the data.** No logged event breaks either rule, even though 3.6% of all
  user/product pairs would fail the income rule and low-risk users make up about a third of users:
  * `min_income`: monthly income must be at least the product's minimum.
  * `risk_suitability`: high-risk products are not offered to low-risk profiles. (Medium-risk
    users *do* engage with high-risk products, so a stricter rule would contradict the data.
    Instead it shows as a *caution*.)
* **Missing data is never "eligible".** If income or risk profile is unknown, any rule needing
  them fails with an explicit reason.
* **Behavior barely depends on the profile.** The apply rate is about 9.4% for every
  employment type, risk profile, age band and income quartile. So no ranker can beat chance
  here, and the evaluation includes a random baseline to make that visible.

## Results

Temporal holdout: train on the earliest 80% of events, then rank the whole catalog for each
of the 174 applications in the last 20%. Those test requests had about 18.5 eligible products each (18.4 per user across all 1,200 users); K=5.

| Strategy | Recall@5 | NDCG@5 | Ineligible slots | Requests showing an ineligible product |
| --- | --- | --- | --- | --- |
| Random (eligible only) | 0.270 | 0.137 | 0% | 0% |
| Popularity (eligible only) | 0.264 | 0.145 | 0% | 0% |
| Ranker, **no eligibility layer** | 0.264 | 0.150 | **5.4%** | **23%** |
| **Eligibility → ranker** | 0.270 | 0.153 | 0% | 0% |

* Relevance: every strategy is at chance on this synthetic data. Read these as a correctly
  measured null result, not as evidence the ranker works. The exact tie with random (0.270)
  comes from the seed used: across 20 seeds, random scores 0.20–0.31 and the ranker 0.22–0.29.
  Ranker scores are relative (training used 8 sampled negatives per application), not real
  application probabilities.
* Policy: the rule layer is what matters. The model was trained only on eligible products, yet
  without the rules it still shows an ineligible product to about 1 in 4 users. **A model never
  learns hard rules**, which is the central lesson of the guide.
* Validity: 0 test applications were for products the rules exclude, so the rules match the data.
* Explanations: `python -m src.experiments` re-checks 3,326 displayed reasons for 200 users
  against source data, with 0 unsupported.

## Explanations

Each reason is built by code from one piece of evidence, and the evidence is returned with it:

| Source | Example reason |
| --- | --- |
| `history` | "They viewed this product before", "They engaged with 2 other loan product(s)" |
| `popularity` | "One of the 5 most engaged-with products (#2)" |
| `rule` / `catalog` | "Meets the minimum monthly income of 50,000", "No minimum income requirement" |
| `profile` | "Low risk, within their medium risk profile". If above the profile, shown as a *caution*. |

Model coefficients are shown separately in the UI as "what the model learned". They are not
presented as reasons.

## Using the UI

* **Pick a user.** The samples cover a range of eligibility situations: 14/20 (low risk, low
  income), 17/20, 17/20 and 20/20, plus an unknown visitor.
* **What-if profile:** change income (with presets), risk profile, employment or age. The
  eligibility count, excluded products and ranking update. Overridden fields show a blue dot;
  Reset returns to the dataset profile.
* **Excluded products:** each one lists the rule it failed, with the user's actual values.
* **"What if there were no eligibility layer?"** shows the same model's ranking over every
  product, with ineligible entries in red (experiment 10).
* **Reason tags** show their evidence on hover.
* **Learn more / Apply** send feedback with the request ID and strategy. "Apply" only records
  feedback; nothing is approved.
* **Retrain model** refits on the new events.
* **Project details** (header pill, or `/about`): a static, plain-language overview of the
  project for recruiters and hiring managers: the problem, features, pipeline, results, engineering
  highlights and tech stack. It makes no API calls, so it works even when the backend is down.

Reset: delete `backend/data/new_interactions.csv` and `backend/feedback/`, then run
`python -m src.train` from `backend/`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/v1/financial-products/recommendations/{user_id}?limit=5` | Guide contract, plus `eligibility` (count, excluded products and failed rules), `reasons`/`cautions` with evidence, `request_id`, `model_version`. Optional what-if params: `monthly_income`, `risk_profile`, `employment`, `age_band`, `existing_products`. `include_unfiltered=true` adds the no-rules comparison. |
| GET | `/v1/eligibility/{user_id}/{product_id}` | Check one pair against the rules, with no model involved |
| GET | `/v1/eligibility/rules` | Rules and their version |
| GET | `/v1/financial-products` | Catalog |
| GET | `/v1/profile/options` | Allowed values for the what-if editor (employment, risk profiles, age bands) |
| GET | `/v1/users?search=` / `/v1/users/{user_id}` | User search, profile and history |
| POST | `/v1/financial-products/events` | `recommendation_click` / `view` / `learn` / `apply` feedback |
| GET | `/v1/model` · POST `/v1/model/retrain` | Metadata and evaluation · refit on new events |

Every recommendation request is logged with its request ID, strategy, model version and rules
version. The financial profile is **not** written to the log (data minimization).

## AWS path (guide sections 35–40)

From `backend/`:

```bash
BUCKET=your-bucket
aws s3 cp data/ s3://$BUCKET/project4-financial/data/ --recursive
```

* **Training:** `aws/sagemaker/train.py`, with `backend/` as `source_dir` (so `src/` ships alongside).
  * It writes `ranker.joblib`, `feature_columns.json`, `metadata.json` (with the eligibility
    rules version) and a data snapshot to `SM_MODEL_DIR`. The endpoint never runs a model
    without its preprocessing assumptions.
  * It writes `metrics.json` to the output data dir; copy it to `evaluation/`.
  * Endpoint dependencies are in `aws/sagemaker/requirements.txt`.
  * To test locally (from `backend/`):
    `SM_CHANNEL_TRAINING=data SM_MODEL_DIR=/tmp/model SM_OUTPUT_DATA_DIR=/tmp/out python aws/sagemaker/train.py`
* **Inference:** `aws/sagemaker/inference.py` runs eligibility **inside** the endpoint, before
  ranking. Request `{"user_id", "limit"}`; response
  `{"strategy", "user_id", "rules_version", "recommendations": [{"product_id", "score", "reasons"}]}`.
  If the rules become legally or operationally critical, move them into a dedicated policy service for a
  clearer audit boundary.
* **API:** API Gateway `GET /v1/financial-products/recommendations/{user_id}` → `aws/lambda/handler.py`
  (env `SAGEMAKER_ENDPOINT`; least-privilege role with only `sagemaker:InvokeEndpoint`).
* **Feedback:** `POST /v1/financial-products/events` → `aws/lambda/feedback_handler.py` → Kinesis
  (env `FEEDBACK_STREAM_NAME`). Only the schema's fields are forwarded.
* **Frontend:** `NEXT_PUBLIC_API_URL` = the API Gateway URL. There are no AWS credentials in the
  browser. Add authentication before any real deployment.

Feedback event:

```json
{
  "event_id": "evt-123",
  "request_id": "req-456",
  "user_id": "F00001",
  "product_id": "FP001",
  "event_type": "recommendation_click",
  "recommendation_strategy": "eligibility_then_ranking",
  "timestamp": "2026-09-30T18:10:00Z"
}
```

## Deploy notes

* **Backend** (any Python host, e.g. Render or Railway): set the service's root directory to
  `project4_financial_recommender/backend`, install with `pip install -r requirements.txt`, and
  start with `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. Set `CORS_ORIGINS` to the
  frontend's URL (comma-separated; it defaults to the local dev server on port 3004). `artifacts/` is committed, so the
  API starts without training; if it is missing, the API trains on startup.
* **Frontend** (e.g. Vercel): root directory `project4_financial_recommender/frontend`, with
  `NEXT_PUBLIC_API_URL` set to the backend URL. `/about` is static and needs no backend.
