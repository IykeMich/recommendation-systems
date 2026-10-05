# Project 2 — ShopSmart Adaptive Hybrid Recommender

A transparent **adaptive hybrid recommender** (a weighted hybrid whose weights adapt to the
shopper's history size) over weighted implicit feedback (`view=1, cart=3, purchase=6`),
served by FastAPI and shown in a Next.js + TanStack Query storefront. It blends two signals:

* **Similar products (content):** TF-IDF over category, subcategory, brand and description,
  plus bonuses for a shared subcategory, category or brand.
* **Shoppers like you (collaborative):** item-item cosine similarity over who interacted with what.

The blend adapts to the shopper. With a short history it is mostly similar products (80%),
and it shifts to mostly collaborative (80%) once they have touched about 16 or more products
(`CONTENT_SHARE_*` in `src/config.py`). Every card has a **Why?** tooltip, for example "37 shoppers
who engaged with Doritos (you bought it) also engaged with this". Built from the
*Project 2 — ShopSmart Full Build Guide*.

In code, the model is the class `AdaptiveHybridRecommender`. Both `model_type` and the API
`strategy` are `adaptive_hybrid`. **[docs/hybrid-model-changes.md](docs/hybrid-model-changes.md)**
compares this design with the original item-item-only version.

```
interactions.csv → event weights → user-item matrix → item-item cosine similarity ┐
products.csv → TF-IDF + attribute bonuses → product-product content similarity ────┤
  → adaptive blend → remove seen items → Top-K + reasons → FastAPI → Next.js
  → feedback events → new interactions → retrain
```

## Structure

The project is split into two independently deployable folders: `backend/` (Python API, model, data) and `frontend/` (Next.js).

| Path | What it holds |
| --- | --- |
| `backend/data/` | `products.csv` (598 branded products, 14 categories) plus synthetic `users.csv` and `interactions.csv`; the previous versions are kept in `data/archive/` |
| `backend/scripts/generate_retail_data.py` | Regenerates the synthetic users and interactions for the current `products.csv` (seeded) |
| `backend/notebooks/01_build_recommender.ipynb` | Guided classroom walkthrough, calling the same `src/` functions |
| `backend/src/config.py` | Paths, event weights, Top-K default, hybrid blend and content-similarity settings |
| `backend/src/data.py` | Loading CSVs and new interactions |
| `backend/src/preprocess.py` | Event weighting, user-item matrix, sparsity |
| `backend/src/recommender.py` | `AdaptiveHybridRecommender` (content + item-item, with reasons), `ItemItemRecommender` and the popularity fallback |
| `backend/src/evaluate.py` | Leave-one-positive-out holdout: Recall/Precision/NDCG@K and coverage for the hybrid, item-item only and popularity |
| `backend/src/train.py` | Trains the hybrid, evaluates it and writes `artifacts/` |
| `backend/src/experiments.py` | New-interaction and new-item experiments (guide §14–15) |
| `backend/artifacts/` | `model.joblib`, `item_similarity.npz`, `item_ids.npy`, `model_metadata.json`, `metrics.json` |
| `backend/api/main.py` | FastAPI app |
| `backend/requirements.txt` | Runtime dependencies (what a host installs); `requirements-dev.txt` adds pytest/httpx |
| `backend/aws/sagemaker/` | SageMaker training + inference entry points, and `package_model.py`, which builds the S3 bundle (`model.tar.gz`, `metrics.json`) into `aws/build/` |
| `backend/aws/lambda/` | API Gateway handlers: recommendations → SageMaker, feedback → Kinesis |
| `backend/tests/` | pytest suite for the model and API |
| `frontend/` | Next.js app: shopper picker, product search, shopper panel, recommendation cards with **Why?** tooltips, model panel |
| `frontend/app/about/page.tsx` | **Project details** page at `/about` (linked from the header): a plain-language summary for recruiters and hiring managers, covering the problem, features, pipeline, results and tech stack. It's static, so it works even when the API is down |
| `docs/hybrid-model-changes.md` | Original vs adaptive hybrid design: what changed, why, evidence and trade-offs |
| `docs/command-line-experiments.md` | How to run, read and write the command-line tests and experiments (worked example: new-product cold start) |

## Run it locally

Backend commands run from `project2_shopsmart/backend/`; frontend commands from `project2_shopsmart/frontend/`.

```bash
# 1. Environment (once), from backend/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -c "import pandas, numpy, sklearn, fastapi"   # checkpoint

# 2. Train + evaluate → backend/artifacts/
python -m src.train

# 3. API (terminal 1, in backend/)
uvicorn api.main:app --reload --port 8002
#    Swagger: http://localhost:8002/docs

# 4. Frontend (terminal 2, in frontend/)
npm install
cp .env.example .env.local      # NEXT_PUBLIC_API_URL=http://localhost:8002
npm run dev                     # http://localhost:3002
```

Tests: `python -m pytest -q`. Experiments: `python -m src.experiments` (both from `backend/`).
See [docs/command-line-experiments.md](docs/command-line-experiments.md) for how to read and write them.

Each project has its own ports (Project 1: 8001/3001, Project 2: 8002/3002, Project 3: 8003/3003,
Project 4: 8004/3004, capstone: 8005/3005), so they can all run at the same time.

## End-to-end checks (guide §13)

```bash
curl localhost:8002/health                                      # {"status":"ok"}
curl "localhost:8002/v1/recommendations/user/R00001?limit=5"   # adaptive_hybrid
curl "localhost:8002/v1/recommendations/user/unknown-user"     # popular_fallback
```

## The feedback loop (guide §14, §16–17)

In the UI, pick a shopper and use **View / Add to cart / Buy** on a recommended card, or on any
product found with **Find a product**. Each action is sent to `POST /v1/events`:

* Every event is logged to `backend/feedback/events.jsonl`. Actions on recommended cards carry
  the `recommendation_request_id` of the list they came from. Actions from product search don't,
  so the log separates "acted on a recommendation" from "found it themselves". Recommendation
  impressions are logged there automatically, which keeps "the system recommended it" separate
  from business transactions.
* `view` / `cart` / `purchase` also append to `backend/data/new_interactions.csv`.
  The original CSVs are never modified.

Recommendations do **not** change until you press **Retrain model** in the Model
panel (or call `POST /v1/model/retrain`, or run `python -m src.train`). After
retraining, cards show ▲/▼/NEW rank movement. Try **New guest**: they start on the
popularity fallback, and after a purchase and a retrain they become a known,
personalized shopper.

**Testing across categories:** use **Find a product** to have a shopper interact outside their
usual categories. It searches the **whole catalog**, not just the recommended products, by name,
brand, category or subcategory, with an optional category filter. Press **Search** (or Enter);
an empty search lists all products. For example, with **New guest**, buy a Nike sneaker and a Milo, then retrain. The
recommendations mix Apparel & Footwear with Food & Beverage, and each card's **Why?** tooltip
names the product it came from.

To reset to the original dataset: delete `backend/data/new_interactions.csv` and
`backend/feedback/`, then run `python -m src.train` from `backend/`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/v1/recommendations/user/{user_id}?limit=10` | Guide contract, plus `recommendation_request_id`; each item has `source` (`similar_products` / `shoppers_like_you`) and `reasons` |
| GET | `/v1/products?search=&category=&limit=24` | Catalog search for the UI (name, brand, category, subcategory, `item_id` prefix); also returns all `categories` |
| GET | `/v1/users?search=` | Shopper search for the UI |
| GET | `/v1/users/{user_id}` | Profile, event counts, recent history (`pending` = not trained on yet) |
| POST | `/v1/events` | Feedback: `recommendation_click`, `view`, `cart`, `purchase` |
| GET | `/v1/model` | Metadata, evaluation, pending interaction count |
| POST | `/v1/model/retrain` | Retrain with new interactions and hot-swap the model |

## The data

`products.csv` is the catalog of real brands, which is the input. Shoppers and their behavior are
synthetic, generated by `scripts/generate_retail_data.py` (run from `backend/`) so that they
match that catalog:

* 1,000 shoppers. Each shopper has a preferred category (biased by segment, e.g. family → Food & Beverage / Baby Care).
* About 28k events. About 60% are in 1–2 favourite subcategories, 10% elsewhere in the preferred category, 18% in a second category and 12% random. Each product touch runs through a view → cart (28%) → purchase (50% of carts) funnel.
* All 598 products have history.

Re-run the script whenever `products.csv` changes, then retrain. Interactions refer to products
by ID, so history generated for an old catalog describes the wrong products.

## Results on the supplied data

Leave-one-positive-out (each shopper's latest cart/purchase hidden), K=10, 973 shoppers:

| Metric | Hybrid | Item-item only | Popularity |
| --- | --- | --- | --- |
| Recall@10 | **0.429** | 0.398 | 0.090 |
| NDCG@10 | **0.280** | 0.252 | 0.047 |
| Coverage | 99% | 99% | 3% |

For **new shoppers** (training keeps only each shopper's first 3 products), Recall@10 is
0.246 for the hybrid versus 0.104 for item-item only. This is where content similarity helps
most. The data is synthetic and built to cluster by category, so real behavior is the real test.

## Deliberate choices beyond the guide's code

* **Model weight comes from `event_type`**, not the supplied `event_value`, as the guide requires.
* **Candidates with a zero score are dropped.** Items with no collaborative link to the shopper aren't recommendation-worthy.
* **The similarity diagonal is zeroed.** This doesn't change rankings, since seen items are filtered anyway. It keeps the "Why this?" explanations from citing the item itself.
* **Other sizes of a product the shopper already has stay eligible.** Excluding them was tried. Re-buying a different pack size is common: 46% of held-out purchases were another size of an item already seen. Excluding sizes dropped Recall@10 from 0.43 to 0.13.
* **Explanations count shoppers per item, not "same history".** Hardly any two shoppers share an identical history, so the honest statement is "N shoppers who engaged with X also engaged with this".
* **The holdout removes every event for the hidden (user, item) pair**, so a leftover view can't leak the answer.
* **The guide's API snippet reads `events["weight"]` before weighting the events.** Here the fallback uses weighted events.

## Deploying (Vercel + a Python host)

* **Frontend → Vercel:** import the repo and set **Root Directory** to `project2_shopsmart/frontend`.
  Set the environment variable `NEXT_PUBLIC_API_URL` to the backend's public URL. It is inlined at build time, so redeploy after changing it.
* **Backend → Render, Railway, Fly.io or similar:** set the root directory to `project2_shopsmart/backend`.
  * Build: `pip install -r requirements.txt`
  * Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  * Set `CORS_ORIGINS` to the frontend URL, for example `https://your-app.vercel.app`. Separate several origins with commas.
* **Caveat:** feedback events, `new_interactions.csv` and retrained artifacts are written to local disk.
  On most hosts that disk is wiped on every redeploy or restart, so the demo resets to the committed model.
  The AWS path below replaces those local files with Kinesis/S3.
* **Don't put the backend on Vercel as it is.** Vercel runs Python as serverless functions with a
  read-only filesystem (only `/tmp` is writable, and it isn't kept between requests), and nothing in memory is shared between instances.
  * Product search, shopper search and the model panel would work, because they only read data.
  * `GET /v1/recommendations/...` would **fail**, because it writes an impression to `feedback/events.jsonl`.
  * `POST /v1/events` and **Retrain** would fail or not persist.

  The dependencies (pandas, NumPy, SciPy, scikit-learn) are also close to the function size limit.
  Use Vercel for the frontend and a host with a long-running process for the backend, or the AWS path.

## AWS path (guide §16–22)

**"SageMaker" here means Amazon SageMaker AI.** That is the ML service (training jobs, models,
endpoints), renamed from "Amazon SageMaker" in December 2024. The name "Amazon SageMaker" now refers to
a broader data and AI platform (Unified Studio), which this project doesn't use. The APIs are
unchanged (`boto3` clients `sagemaker` and `sagemaker-runtime`). In the console, look under
**Amazon SageMaker AI**.

**Bucket name.** S3 bucket names are global across all AWS accounts, lowercase, 3–63 characters,
with hyphens and no underscores. One bucket can hold every lab project, with each project under its own
prefix (`shopsmart/` here). Adding your account ID and region makes the name unique and
self-describing:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
BUCKET=recsys-lab-$ACCOUNT_ID-$REGION          # e.g. recsys-lab-123456789012-eu-west-2
aws s3 mb s3://$BUCKET --region $REGION
```

**Build the bundle** (from `backend/`). This trains exactly as SageMaker would and writes the guide §17 layout to
`aws/build/`, which is git-ignored:

```bash
python aws/sagemaker/package_model.py
# aws/build/data/{users,products,interactions}.csv
# aws/build/code/sourcedir.tar.gz    train.py + inference.py + src/, for a SageMaker AI training job
# aws/build/models/model.tar.gz      model.joblib, popular_items.json, model_metadata.json + code/
# aws/build/evaluation/metrics.json
```

The `code/` prefix is an addition to the guide's §17 layout. A training job runs its code from S3, so
in the console set the hyperparameters `sagemaker_program = train.py` and
`sagemaker_submit_directory = s3://$BUCKET/shopsmart/code/sourcedir.tar.gz`. The archive has no
`requirements.txt` on purpose: the container's own libraries train the model, so it loads in the same
image used for serving.

**Upload** (from `backend/`). Copy only those files. Don't upload `data/` itself, because it also holds
`new_interactions.csv` and `archive/`:

```bash
aws s3 cp aws/build/ s3://$BUCKET/shopsmart/ --recursive
aws s3 ls s3://$BUCKET/shopsmart/ --recursive
```

`model.tar.gz` contains `code/inference.py`, `code/src/` and `code/requirements.txt`, which is pinned to
the pandas/NumPy/SciPy/scikit-learn versions that built the model. The saved model is a Python
pickle, so it only loads under compatible versions. If the serving container can't install those
versions, run `train.py` as a SageMaker training job instead (below), so the same container
trains and serves the model.

* **Training:** run `aws/sagemaker/train.py` as the entry point, with `backend/` as `source_dir` so
  `src/` ships alongside it, and the `training` channel set to `s3://$BUCKET/shopsmart/data/`.
  It writes `model.joblib`, `popular_items.json` and metadata to `SM_MODEL_DIR`, and
  `metrics.json` to the output data dir (copy it to `s3://$BUCKET/shopsmart/evaluation/`).
  To test locally:
  `SM_CHANNEL_TRAINING=data SM_MODEL_DIR=/tmp/model SM_OUTPUT_DATA_DIR=/tmp/out python aws/sagemaker/train.py`
* **Inference:** `aws/sagemaker/inference.py` (`model_fn` / `input_fn` / `predict_fn` / `output_fn`).
  Input `{"user_id", "limit"}`; output `{"user_id", "strategy", "recommendations": [{"item_id", "score", "source"}]}`.
  `train.py` needs both `interactions.csv` and `products.csv` in the training channel.
  The model is loaded once per container. `src/` must be packaged with it, because the pickled
  model references `src.recommender`.
* **Recommendation API:** API Gateway `GET /v1/recommendations/user/{user_id}` → `aws/lambda/handler.py`
  (env `SAGEMAKER_ENDPOINT`; role needs `sagemaker:InvokeEndpoint`). Product metadata is joined
  in the API layer, not the endpoint.
* **Feedback:** API Gateway `POST /v1/events` → `aws/lambda/feedback_handler.py` → Kinesis
  (env `FEEDBACK_STREAM_NAME`; role needs `kinesis:PutRecord`) → S3 → Glue/Athena → retraining.
* **Frontend:** set `NEXT_PUBLIC_API_URL` to the API Gateway URL. The browser never holds AWS credentials.
  The two Lambda handlers cover only recommendations and events. Shopper search and profiles,
  product search and the model panel are FastAPI-only for now, so they need routes of their own
  (or the FastAPI app behind API Gateway) before the full UI runs on AWS.

Feedback event schema:

```json
{
  "event_id": "evt-123",
  "user_id": "R00001",
  "item_id": "P00123",
  "event_type": "recommendation_click",
  "recommendation_request_id": "req-456",
  "timestamp": "2026-09-30T12:00:00Z"
}
```
