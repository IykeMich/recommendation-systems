# Capstone — Fraud Detection & Prevention

A fraud-risk platform with separate layers for data quality, features, model,
policy, API and operations dashboard, plus an AWS design (S3, Kinesis, Lambda,
SageMaker, API Gateway, DynamoDB, CloudWatch). Built from the *Capstone — Fraud
Detection & Prevention Full Build Guide*.

> **Synthetic data, learning-only label.** This is not a live financial control
> system, and its metrics say nothing about real-world fraud prevalence or
> detection performance.

```
DATA → TRUST (quality checks) → FEATURES → MODEL → RISK SCORE (calibrated)
     → POLICY (ALLOW / REVIEW / BLOCK) → DECISION (idempotent, versioned)
     → API → DASHBOARD → ANALYST OUTCOME → MONITORING → RETRAINING
```

**Model = estimates risk. Policy = decides.** The thresholds live in
`backend/artifacts/policy.json`. They can be changed from the dashboard or `PUT /v1/policy`
without retraining, and every decision records the model and policy version it was made under.

## Run it

All Python commands run from `backend/`:

```bash
cd capstone_fraud_detection/backend
python3 -m venv .venv && source .venv/bin/activate      # once
pip install -r requirements-dev.txt                      # once (runtime deps + pytest/httpx)
python -c "import pandas, sklearn, fastapi, joblib; print('environment OK')"

python -m src.train          # quality checks, split, 4 models + rule reference, calibration → artifacts/
python -m src.experiments    # the guide's experiments → artifacts/experiment_log.csv
python -m pytest -q          # 39 tests

uvicorn api.main:app --reload --port 8005                # terminal 1 (Swagger: /docs)
cd ../frontend && npm install && cp .env.example .env.local && npm run dev   # terminal 2 → http://localhost:3005
```

The dashboard's header has a **Project details** link to `/about`: a static, plain-language
overview of the project (problem, features, pipeline, test-set results, tech stack) for
recruiters and hiring managers. It needs no API, so it works even when the backend is down.

Each project has its own ports (Project 1: 8001/3001, Project 2: 8002/3002, Project 3: 8003/3003,
Project 4: 8004/3004, capstone: 8005/3005), so they can all run at the same time.

In the dashboard, click **Send next 250** under *Stream test transactions*. That pushes
held-out transactions through the scoring path. Then open alerts, record outcomes and try the policy sliders.

## Structure

Two top-level folders: `backend/` (everything Python: data, model, API, AWS code, tests) and
`frontend/` (the Next.js dashboard). Paths below are relative to `backend/` unless they start with `frontend/`.

| Path | What it holds |
| --- | --- |
| `data/raw/transactions.csv` | 20,000 synthetic transactions (the capstone fraud dataset) |
| `data/processed/` | `train.csv`, `validation.csv`, `test.csv` written by training |
| `data/predictions/` | Test-set scores and decisions |
| `notebooks/01…05` | Inspect → features → train → thresholds → explanations. All call `src/`. |
| `src/schema.py` | Alias-based schema adapter (fails loudly) and data-quality checks |
| `src/features.py` | Leakage-safe features and customer reference profiles |
| `src/train.py` | Split, model comparison, validation-based selection, calibration, artifacts |
| `src/evaluate.py` | PR-AUC, ROC-AUC, Brier, precision/recall/F1, confusion matrix, threshold table |
| `src/policy.py` | `Policy` / `decide`: the only place ALLOW/REVIEW/BLOCK is decided |
| `src/explain.py` | Evidence-backed reasons, separate from the model |
| `src/predict.py` | `FraudScorer`, shared by the API, SageMaker and batch scoring |
| `src/experiments.py` | Experiments E01–E06 (E07/E08 are API tests) |
| `artifacts/` | `fraud_pipeline.joblib`, `feature_manifest.json`, `metrics.json`, `threshold_table.csv`, `policy.json`, `data_quality.json`, `experiment_log.csv` |
| `api/main.py`, `api/store.py` | FastAPI app (`api.main:app`) and the SQLite decision store (idempotency, outcomes, rejections) |
| `requirements.txt`, `requirements-dev.txt` | Runtime dependencies; dev adds pytest and httpx |
| `frontend/` | Next.js + TanStack Query operations dashboard |
| `frontend/app/about/page.tsx` | The static **Project details** page (`/about`) |
| `aws/sagemaker/` | `inference.py`, `package.sh` (builds `model.tar.gz`) |
| `aws/lambda/` | `score_api.py` (API Gateway), `stream_consumer.py` (Kinesis), `fraud_common.py` |
| `aws/sam/template.yaml` | HTTP API, Kinesis stream, both Lambdas, DynamoDB, log retention, alarm, least-privilege IAM |
| `tests/` | Features, policy, API, and AWS handlers (with in-memory fakes) |

## What the data is (inspected first)

| Guide contract | This dataset |
| --- | --- |
| `transaction_timestamp` | **Missing.** Only `hour` (0–23). |
| `customer_id` | `user_id` (the same 1,200 users as Project 4) |
| `merchant_id`, `device_id` | **Missing.** Only `merchant_category` and `device` type (android/ios/web). |
| `location` / `channel` | `country` (NG, GH, GB, US) |
| velocity | `velocity_1h`, already computed (1–12) |
| `is_fraud` | `fraud_label`: 1,557 fraud (7.8%) |

Data quality: 0 duplicates, 0 missing values, 0 invalid amounts, hours or labels. One currency (NGN).

**The label follows one synthetic rule.** Fraud is 71% when `hour` is 1–4 **and**
`velocity_1h` ≥ 6, and about 1% everywhere else. Amount, country, device and
merchant category have no effect. 89% of fraud falls in that region; the other 11%
(175 cases) is noise no model can find. `transaction_id` order shows no time trend.

### What that changed

* **No temporal split is possible.** There are no timestamps, and the guide's own check
  shows row order is not time: splitting by `transaction_id` order gives the same
  PR-AUC as random (0.703 vs 0.700). The split is **stratified random 70/15/15, seeded**.
  A split by unseen customers (0.662) is also reported.
* **"History" = the training split.** Customer profiles (count, average amount, device and
  country shares) come from training rows only. Each training row uses **leave-one-out**,
  so it never describes itself. Validation, test and live requests use the training profiles.
* **Novelty is per device type and country,** not per device ID or merchant ID, because those don't exist.

## Results (test set, 3,001 transactions, 234 fraud)

Selection used **validation PR-AUC**; the test set was scored once, afterwards.
Recall and precision are at the review threshold of 0.55.

| Model | PR-AUC | ROC-AUC | Brier | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| Logistic regression (guide baseline, hour as a number) | 0.440 | 0.911 | 0.117 | 91% | 34% |
| Logistic regression, hour as a category | 0.657 | 0.934 | 0.083 | 90% | 42% |
| Random forest | 0.700 | 0.935 | 0.040 | 88% | 69% |
| Gradient boosting (selected on validation) | 0.665 | 0.942 | 0.044 | 90% | 70% |
| **Gradient boosting, calibrated (served)** | 0.635 | 0.944 | **0.030** | 89% | 69% |
| *Synthetic rule, reference ceiling* | *0.632* | *0.932* | *0.039* | *90%* | *70%* |

How to read this:

* **The strong models sit at the rule's ceiling.** This is the guide's troubleshooting
  row "scores almost perfect → synthetic shortcut". The differences between them are noise
  on 234 fraud cases. The honest claim is that the pipeline finds the planted pattern,
  not that the model is good at fraud.
* **Why logistic regression lost:** a linear model can't express "hours 1–4" from a
  numeric hour. Treating hour as a category fixes most of the gap. That is the guide's
  "justify model complexity" experiment.
* **Why the served model is calibrated.** `class_weight="balanced"` (the guide's setting)
  pushes suspicious scores to about 0.95. That sends every one of them to **BLOCK** and
  none to REVIEW (test mix: 0% review, 10% block). Isotonic calibration on the training
  split makes a score mean "this share of similar transactions were fraud". The riskiest
  pattern is only about 71% fraud, so under policy-v1 (0.55 / 0.85) suspicious
  transactions go to **REVIEW** (10% of the test set) and **nothing is auto-blocked**.
  That's the defensible outcome for this data. To block, lower the BLOCK threshold
  deliberately. The dashboard's policy simulator shows the cost on validation data first.

### Experiments (`artifacts/experiment_log.csv`)

| # | Change | Result |
| --- | --- | --- |
| E01 | Remove velocity | PR-AUC 0.700 → **0.455**; precision 69% → 42% |
| E02 | Remove device/country novelty | 0.700 → 0.681. Almost no value in this data. |
| E03 | Change review threshold only | Model fixed. Recall 90% at 0.30, 89% at 0.55, 74% at 0.70, 20% at 0.75. |
| E04 | Split design | Random 0.700 · `transaction_id` order 0.703 · unseen customers 0.662 |
| E05 | Balanced vs unbalanced weights | PR-AUC 0.700 vs 0.661. Balanced gives 9.4% BLOCK at 0.85; unbalanced gives 0%. |
| E06 | Logistic vs random forest | 0.440 (numeric hour) / 0.657 (categorical hour) / 0.700 |
| E07 | Retry the same transaction | Stored decision returned, no duplicate alert. A different payload with the same ID returns 409. |
| E08 | Malformed records | Rejected with 422 and counted in the data-quality panel |

## Explanations

A reason is shown only when the transaction's bucket (hour, velocity, amount range,
country, device, category) had a training fraud rate at least **1.5× the overall rate**.
The response includes that evidence, e.g. *"Transaction at 3:00…, 41% fraud in
training vs 7.8% overall, 5.3×"*. Novelty facts ("device type new for this customer",
"amount 4× their average") are true, but carry no signal in this data. They are
returned as **context**, not reasons, so the system never claims a signal it doesn't use.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Status, model version, policy version |
| POST | `/score` | Score one transaction. Validation, idempotency, features, score, policy, reasons. |
| POST | `/v1/replay` | Stream held-out test transactions through `/score`'s logic (simulated Kinesis) |
| GET | `/v1/decisions` · `/v1/decisions/{id}` | Recent decisions · alert detail with features, evidence and customer profile |
| POST | `/v1/decisions/{id}/outcome` | Analyst outcome: `confirmed_fraud` or `legitimate` |
| GET | `/v1/stats` · `/v1/data-quality` | KPIs, labelled precision, rejected requests |
| GET | `/v1/model` | Manifest, metrics, threshold table, training data quality |
| GET/PUT | `/v1/policy` | Read or change thresholds (new version; affects new decisions only) |
| GET | `/v1/policy/simulate` | Preview a policy on validation data before applying it |

The browser is untrusted, so the server:
* rejects unknown fields (a client can't send its own `risk_score`);
* computes features and the score itself;
* computes `velocity_1h` from the decision store when a `transaction_timestamp` is given.

Unknown countries, devices and categories don't crash the model.
**Protect `PUT /v1/policy` and the outcome endpoint with authentication in any real deployment.**

## AWS path

```
Producer → Kinesis → stream_consumer Lambda ─┐
Browser → API Gateway → score_api Lambda ────┼→ SageMaker endpoint (risk score)
                                             │→ policy (Lambda env thresholds)
                                             └→ DynamoDB conditional put (idempotency) → CloudWatch logs
```

Run these from `backend/` (the `aws/` folder lives there).

1. `aws/sagemaker/package.sh s3://YOUR-BUCKET/models/fraud/v1/` builds `backend/model.tar.gz`
   (artifact, manifest, `inference.py` and `src/`) and uploads it.
2. Create a SageMaker model with the scikit-learn inference container, pointing at it.
   Then create an endpoint config and an endpoint.
3. `cd aws/sam && sam build && sam deploy --guided`, passing `SageMakerEndpointName`.
   This creates the HTTP API, the Kinesis stream, both Lambdas, the DynamoDB table,
   14-day log retention and an error alarm.
   * IAM is least-privilege: `sagemaker:InvokeEndpoint` on one endpoint, and CRUD on one table.
   * The endpoint returns only a score. The Lambdas apply the policy from environment
     variables, so thresholds change without redeploying the model.
   * The stream consumer skips malformed records and reports transient failures as
     batch-item failures. Idempotency makes redelivery safe.
4. Point the dashboard's `NEXT_PUBLIC_API_URL` at the API output.

The Lambdas and the SageMaker handler are tested locally in `backend/tests/test_aws_handlers.py`.
**Nothing here has been deployed.**

### Hosting the app elsewhere (e.g. a PaaS)

* Backend: set the service's root directory to `capstone_fraud_detection/backend`, install with
  `pip install -r requirements.txt` on Python 3.11, and start with `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
  On Render, the repo's `render.yaml` defines this service as `fraud-detection-api`.
  `requirements.txt` pins the versions `artifacts/fraud_pipeline.joblib` was trained with; retrain
  after upgrading scikit-learn or numpy. Localhost on any port and any `https://*.vercel.app` URL are always allowed; set `CORS_ORIGINS` (comma-separated) only for a custom dashboard domain. The SQLite store (`data/decisions.sqlite3`) is local to
  the instance, so it resets on redeploy unless you attach a disk.
* Frontend: root directory `capstone_fraud_detection/frontend`, with `NEXT_PUBLIC_API_URL` pointing at the backend.

### Teardown (every lab session)

`sam delete`, then delete the SageMaker endpoint, endpoint config and model, and remove
`model.tar.gz` and data from S3. Review CloudWatch log groups and your billing dashboard.
SageMaker endpoints are charged while they run.

## Limitations

* Synthetic, near-deterministic label. Metrics measure the pipeline, not fraud skill.
* No timestamps: no temporal split, no true "prior history". Profiles are a static training-period reference.
* No merchant or device identifiers: novelty is only at the device-type and country level.
* Customer profiles don't update online after training. An online store (e.g. DynamoDB) is the extension.
* Only 234 fraud cases in the test set, so differences of a few PR-AUC points are noise.
* The local SQLite store is single-node; the AWS design uses DynamoDB.
