# Project 3 — Context-Aware Offer Recommender

Project 2 asked *"what products fit this user?"* Project 3 asks *"which offer fits
this user **right now**?"*, given their history, the offer's attributes and the
current context (device, region, hour, day). Built from the *Project 3
Context-Aware Offer Recommender Full Build Guide*.

```
user profile + offer history + offer catalog + current context
  → point-in-time features → candidate generation (business rules)
  → logistic-regression ranker → Top-K → FastAPI → Next.js
  → feedback (with context) → history features update now, weights on retrain
```

## Structure

The Python side lives in `backend/`; the Next.js app lives in `frontend/`.

| Path | What it holds |
| --- | --- |
| `backend/data/` | `users.csv` (a copy of Project 2's current 1,000 shoppers; same user IDs), plus synthetic `offers.csv` (120 offers) and `offer_interactions.csv` (6,961 events) generated to match them. The previous versions are kept in `data/archive/` |
| `backend/scripts/generate_offer_data.py` | Regenerates the synthetic offers and interactions for the current `users.csv` (seeded) |
| `backend/notebooks/01_contextual_recommender.ipynb` | Guided walkthrough calling the same `src/` code |
| `backend/src/config.py` | Paths, column adapter, label rule, sampling and split settings |
| `backend/src/data.py` | Loading and the raw → internal schema adapter |
| `backend/src/features.py` | Time/context encodings, point-in-time user and offer features, training examples |
| `backend/src/candidates.py` | Eligibility rules with a per-step funnel |
| `backend/src/model.py` | The sklearn pipeline, coefficients, per-offer reasons |
| `backend/src/recommender.py` | `ContextualRecommender`: candidates → rank → Top-K, plus cold start |
| `backend/src/evaluate.py` | Time-aware holdout: Recall/NDCG@K, coverage, diversity, per-segment |
| `backend/src/train.py` | Trains, evaluates and writes `artifacts/` |
| `backend/src/experiments.py` | The 8 experiments from guide section 35 |
| `backend/artifacts/` | `ranker.joblib`, `feature_columns.json`, `metadata.json`, `metrics.json` |
| `backend/api/main.py` | FastAPI app |
| `backend/aws/sagemaker/`, `backend/aws/lambda/` | SageMaker train/inference, API Gateway handlers (inference and Kinesis feedback) |
| `backend/tests/` | pytest suite |
| `backend/feedback/` | Runtime feedback log (`events.jsonl`, git-ignored) |
| `backend/requirements.txt`, `backend/requirements-dev.txt` | API/runtime dependencies; dev adds pytest and httpx |
| `frontend/` | Next.js + TanStack Query app |
| `frontend/app/about/page.tsx` | **Project details** page (`/about`): a static, plain-language summary for portfolio visitors |

## Run it locally

Backend commands run from `project3_context_offers/backend/`:

```bash
cd backend

# 1. Environment (once)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Train + evaluate → artifacts/
python -m src.train

# 3. API (terminal 1). Swagger: http://localhost:8003/docs
uvicorn api.main:app --reload --port 8003

# 4. Frontend (terminal 2, from project3_context_offers/)
cd frontend
npm install
cp .env.example .env.local      # NEXT_PUBLIC_API_URL=http://localhost:8003
npm run dev                     # http://localhost:3003
```

Tests (from `backend/`): `python -m pytest -q`. Experiments: `python -m src.experiments`.

The header's **Project details** pill opens `/about`, a static overview of the project (problem,
features, pipeline, results, stack) for recruiters and reviewers. It makes no API calls, so it works
even when the backend is down.

Each project has its own ports (Project 1: 8001/3001, Project 2: 8002/3002, Project 3: 8003/3003,
Project 4: 8004/3004, capstone: 8005/3005), so they can all run at the same time.

## What the data actually says (read this first)

Step 03 of the guide says to inspect the data before modeling. Doing so changed three things:

1. **There is no `accepted` label.** Events are `impression`, `click` and `redeem`.
   Label rule: click or redeem = positive. An impression alone is not treated as a
   rejection, because the user may not have noticed the offer.
2. **Whether a shown offer gets clicked barely depends on anything.** Engagement
   rates are flat (about 45%) across day, device, region, discount and channel, and
   move only between about 37% and 52% across the 24 hours. The real signal is
   *which* offers people engage with: about 80% of each user's events are in their
   `preferred_category` (from `users.csv`).
3. **So the guide's literal setup ranks poorly here.** Its setup trains on logged
   rows labelled engaged / not engaged, using only user stats and context. With no
   offer features, every offer gets the same score for a request. Even with offer
   features added, it only reaches Recall@10 = 0.065. The model here uses
   click/redeem positives against 4 sampled offers in the same context, which is
   standard for implicit feedback.

All user and offer history features are **point-in-time**: they count only events
before the request. Training, evaluation and the API share `build_feature_frame`,
so there is no training/serving skew (guide section 26).

## Results

Time-aware holdout: train on the earliest 80% of events, then rank all 120 offers
for each of the 601 clicks/redeems in the last 20%.

| Model | Recall@10 | NDCG@10 | Coverage | Diversity |
| --- | --- | --- | --- | --- |
| Contextual ranker | 0.636 | 0.315 | 93% | 0.17 |
| Without context features (exp. 7) | 0.659 | 0.325 | 88% | 0.16 |
| Guide-literal exposure labels | 0.065 | 0.027 | 65% | 0.25 |
| Popularity baseline (exp. 6) | 0.095 | 0.050 | 11% | 0.45 |
| Contextual, max 3 per category | 0.260 | 0.178 | 55% | 0.42 |

What the results show:
* The ranker beats popularity by about 6.7×.
* **Context adds no measurable lift on this dataset.** The guide warns this can
  happen ("do not assume the contextual model will improve every metric"). The
  per-segment table in `artifacts/metrics.json` shows the same in every daypart and
  device.
* Context still changes what a shopper sees. Region and device rules change what
  is eligible: moving a shopper from Lagos to Abuja keeps only 4 or 5 of their
  Top-10 (measured on 50 shoppers). The hour changes the list too: from 9:00 to
  21:00, every one of those 50 shoppers had between 1 and 3 offers replaced. That
  comes from the category × daypart and category × device features, which with 14
  categories are learned from few examples each. The offline test shows they add
  no accuracy, so treat these hour effects as noise the model picked up, not as
  real shopper behaviour.
* Capping 3 offers per category raises diversity from 0.17 to 0.42 but costs most
  of the recall. That is a real product trade-off, measured here.

### The data

`users.csv` is a copy of Project 2's current shoppers (Project 2 regenerated them in
October 2026 with 14 product categories). The offers and interactions are synthetic,
generated by `scripts/generate_offer_data.py` (run from `backend/`) to match those
shoppers, with the same design as the original Project 3 offer dataset:

* 120 offers across the shoppers' 14 preferred categories, with more offers in
  categories more shoppers prefer (17 Electronics down to 5 Pet Care). Discount
  (5–30%), minimum spend, channel, region (Lagos, Abuja, Port Harcourt or `all`)
  and expiry (24 September to 31 December 2026) are random.
* 6,961 events across 2025, 4–10 per shopper: 3,818 impressions, 2,478 clicks and
  665 redeems. About 80% are offers in the shopper's preferred category. Event type
  is random, which is why engagement is flat.
* Each event uses the shopper's own device and region, but the log ignores offer
  rules: only 38% of logged events are offers available in that region, and 65% are
  offers built for that device. The live API applies the rules; the history does not.

If Project 2's `users.csv` changes again, copy it into `backend/data/`, re-run the
script, then `python -m src.train`. Interactions refer to offers by ID and the
offers' categories come from the shoppers, so old offers and history no longer fit.

## Using the UI

* **Pick a shopper.** The context defaults to their usual device and region.
* **Change the context** with the device, region, hour slider, day and "use current time"
  controls. Every context value is in the TanStack Query key.
* **Pin this context**, then change something. Cards show ▲/▼/NEW against the pinned
  list, with a count of how many offers are shared.
* **Rules**: toggle the region and device eligibility rules, and the 3-per-category cap.
  The funnel shows how many offers each rule removed.
* **Click / Redeem** send feedback with the context to `POST /v1/offers/events`.
  * History features update immediately: a redeemed offer disappears, and category
    affinity shifts.
  * The model's weights change only when you press **Retrain model**. Retraining keeps
    the last offline evaluation; re-run `python -m src.train` to re-evaluate.
* **Unknown visitor** gets `cold_start_popular`: eligible offers ranked by popularity.

To reset: delete `backend/data/new_offer_interactions.csv` and `backend/feedback/`, then run
`python -m src.train` from `backend/`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/v1/offers/recommendations/{user_id}?device_type=&region=&hour=&day_of_week=&limit=` | Guide contract, plus `request_id`, `candidate_funnel` and per-offer `reasons`. Optional `enforce_region`, `enforce_channel`, `max_per_category` |
| GET | `/v1/users?search=` / `/v1/users/{user_id}` | Shopper search, profile and offer history |
| POST | `/v1/offers/events` | `click` / `redeem` feedback with its context |
| GET | `/v1/model` | Metadata, evaluation, readable coefficients, pending feedback |
| POST | `/v1/model/retrain` | Refit the ranker on all interactions |
| GET | `/v1/context/options` | Device and region values for the UI |

## AWS path (guide sections 27–32)

Run these from `backend/`:

```bash
BUCKET=your-bucket
aws s3 cp data/ s3://$BUCKET/project3-context-offers/data/ --recursive
```

* **Training:** `aws/sagemaker/train.py`, with `backend/` as `source_dir` so `src/` ships
  with it. The training channel needs `offers.csv`, `offer_interactions.csv` and `users.csv`.
  * It writes `ranker.joblib` and `feature_columns.json`, plus a data snapshot the endpoint
    needs for history features (production would use a feature store), to `SM_MODEL_DIR`.
  * It writes `metrics.json` to the output data dir; copy it to `evaluation/`.
  * To test locally:
    `SM_CHANNEL_TRAINING=data SM_MODEL_DIR=/tmp/model SM_OUTPUT_DATA_DIR=/tmp/out python aws/sagemaker/train.py`
* **Inference:** `aws/sagemaker/inference.py`. Request `{"user_id", "context": {device_type, region, hour,
  day_of_week}, "limit"}`; response `{"user_id", "strategy", "context", "recommendations": [{"offer_id", "score"}]}`.
* **API:** API Gateway `GET /v1/offers/recommendations/{user_id}` → `aws/lambda/handler.py`
  (env `SAGEMAKER_ENDPOINT`), which turns query parameters into the inference body.
* **Feedback:** `POST /v1/offers/events` → `aws/lambda/feedback_handler.py` → Kinesis
  (env `FEEDBACK_STREAM_NAME`) → S3 `feedback/events/` → Glue/Athena → retraining.
* **Frontend:** set `NEXT_PUBLIC_API_URL` to the API Gateway URL. No AWS credentials in the browser.
* **Hosting the API elsewhere** (Render, Railway, Fly, etc.): set the service's root directory to
  `project3_context_offers/backend`, install with `pip install -r requirements.txt` and start with
  `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. The frontend's root directory is
  `project3_context_offers/frontend`.

Feedback event, which always carries the context that produced the recommendation:

```json
{
  "event_id": "evt-001",
  "request_id": "req-001",
  "user_id": "R00001",
  "offer_id": "O00012",
  "event_type": "click",
  "context": {"device_type": "mobile", "region": "Lagos", "hour": 18, "day_of_week": 2},
  "timestamp": "2026-09-30T18:10:00Z"
}
```
