# ShopSmart: original design vs Adaptive Hybrid

*Change record, 1 October 2026. It covers everything that differs from the original
item-item-only ShopSmart, why each change was made, and the evidence behind it.*

## Summary

ShopSmart originally recommended products with **item-item collaborative filtering** only:
"shoppers who interacted with what you interacted with also interacted with this". It is now an
**Adaptive Hybrid Recommender**. It blends that collaborative signal with **content similarity**
(products that share a subcategory, category, brand or description). The mix depends on how
much history a shopper has.

The change was triggered by a bug report: a new shopper who only bought Food & Beverage items
was recommended sneakers, phones and furniture. The root cause was in the **data**, not the
algorithm. Fixing the data alone would have fixed that report. The hybrid was added because it
also helps shoppers with short histories, and the measurements below back that up.

| | Original | Now |
|---|---|---|
| Model name | Item-item cosine (`item_item_cosine`) | Adaptive Hybrid (`AdaptiveHybridRecommender`, `adaptive_hybrid`) |
| Signals | Who-bought-what only | Who-bought-what **+** product attributes |
| New shopper (few interactions) | Weak, noisy collaborative signal | Mostly similar products, shifting to collaborative as history grows |
| Products nobody has touched | Can never be recommended | Reachable through content similarity |
| Explanation on card | "Because they liked X and Y" (inline) | **Why?** tooltip: source plus reasons with real shopper counts |
| Product discovery in UI | Only via recommended cards | **Find a product** search across the whole catalog |

## 1. Why it changed

### The data stopped matching the catalog

- `products.csv` was replaced: from 500 synthetic products in 6 categories ("Kora Shirts 1"…) to
  **598 real branded products in 14 categories and 121 subcategories**.
- `interactions.csv` and `users.csv` were **not** regenerated. Interactions point at products by ID,
  so a row that used to mean "bought a laptop" now meant "bought Doritos".
- Measured effect: under the previous catalog, **85%** of a shopper's activity was in one category.
  Under the new catalog, the same rows spread out to **32%**, which is close to random.
- Products P00501–P00598 had **no history at all**, so collaborative filtering could never
  recommend them.

### The symptom

`GUEST-001` bought only Food & Beverage (Activia, Ritz, Pop-Tarts, Doritos and so on). After a
retrain, their top 12 were Nike Air Jordans, an iPhone 16 Pro, an IKEA chair, Chanel No. 5,
LEGO and so on. The model was working as designed; it had learned from mismatched data.

## 2. What changed

### 2.1 Data (regenerated for the new catalog)

- New script: `backend/scripts/generate_retail_data.py`. It is seeded, so it gives the same output every run.
- **products.csv is untouched.** It is the input.
- New `users.csv`: 1,000 shoppers (`R00001`–`R01000`, same columns). Preferred categories are drawn from
  the 14 new categories and nudged by segment (e.g. *family* → Food & Beverage, Baby Care).
- New `interactions.csv`: 27,811 events in 2025, same 7 columns. Each shopper spends about 60% of
  their activity in 1–2 favourite subcategories, 10% elsewhere in their preferred category, 18% in a
  second category and 12% exploring at random. Each product they touch goes through view → cart (28%)
  → purchase (50% of carts).
- Result: top-category share is **70%**, every one of the 598 products has history (at least 3 shoppers,
  median 24), and the 19.6k / 5.4k / 2.7k view/cart/purchase split is close to the original's.
- The previous files are kept in `backend/data/archive/`.

### 2.2 Model

| Part | How it works |
|---|---|
| Collaborative ("shoppers like you") | Unchanged item-item cosine over the weighted user × item matrix |
| Content ("similar products") | TF-IDF over category + subcategory + brand + description (with the "Variant: …" suffix removed), plus bonuses for the same subcategory (+0.5), category (+0.2) and brand (+0.1) |
| Blend | Both scores are scaled to 0–1. `content_share = clip(1 − products_seen / 20, 0.2, 0.8)`, so 3 products seen → 80% content, 10 → 50%, 16+ → 20% |
| Filtering | Products the shopper already interacted with are removed. Other sizes of them are **kept** (see §4) |
| Source label | Each result is labelled by whichever part contributed more: `similar_products` or `shoppers_like_you` |

### 2.3 Explanations

Each recommendation now carries `reasons`. Each reason names one of the shopper's own products and
their strongest action on it (bought > carted > viewed):

- **similar_product:** "Similar to *Kinder Bueno* (you bought it), with the same subcategory: Chocolate & Confectionery"
- **co_interaction:** "*37* shoppers who engaged with *Doritos* (you bought it) also engaged with this"

The shopper counts are real: the number of training shoppers who interacted with **both** products.
The original request was for "N users with the same history". We count per product instead, because
almost no two shoppers have an identical history, so that number would nearly always be 0 or 1.

### 2.4 API (`backend/api/main.py`)

| | Original | Now |
|---|---|---|
| Strategy for known shoppers | `collaborative_item_item` | `adaptive_hybrid` |
| Fields on each recommendation | `because: [{item_id, name}]` | `source`, plus `reasons: [{type, item_id, name, your_event, shopper_count \| shared + value}]` |
| Product search | — | `GET /v1/products?search=&category=&limit=` returns `{total, categories, products}` |
| Old saved models | — | If `artifacts/model.joblib` holds an older model type, the API retrains on start |

### 2.5 UI (`frontend/`)

- **Why? tooltip:** the source label and the top two reasons appear on hover, or on focus or tap, instead of taking up space on the card.
- **Find a product:** a search panel over the **whole catalog**, not just recommended products. It
  has a text box, a category filter and a **Search** button (or Enter; an empty search lists all
  products), plus View / Add to cart / Buy, acting as the selected shopper. Actions from search are logged **without** a
  `recommendation_request_id`, so analysis can tell "acted on a recommendation" apart from
  "found it themselves".
- **Model panel:** shows Hybrid, Item-item and Popular side by side, with the best value in each row highlighted.
- Header tagline and strategy description now describe the adaptive hybrid.

### 2.6 AWS scripts (`backend/aws/sagemaker/`)

- `train.py` builds the hybrid and now needs **`products.csv` as well as `interactions.csv`** in the
  training channel. The README's `aws s3 cp data/ …` already uploads both.
- `inference.py` reports `strategy: "adaptive_hybrid"` and each result includes `source`.
- New `package_model.py` builds the guide §17 S3 bundle locally in `aws/build/`:
  - `code/sourcedir.tar.gz`: training code for a SageMaker AI training job (`train.py`, `inference.py`, `src/`)
  - `models/model.tar.gz`: a locally trained model, plus `code/` with `inference.py`, `src/` and a pinned `requirements.txt`
  - `evaluation/metrics.json`
- Lambda handlers are unchanged. Product search, like shopper search, has no Lambda route yet.

### 2.7 Tests

13 → **20** tests, all passing. New ones cover:
- content similarity ranking the same subcategory above unrelated products
- the adaptive share at the low and high ends
- reaching a product nobody has interacted with
- the co-interaction counts
- the API response shape
- product search
- regression: a Food & Beverage-only shopper gets at least 70% Food & Beverage
- a two-category shopper gets both categories

## 3. Evidence

Protocol (unchanged): for each shopper, hide their most recent cart or purchase, train on the
rest, and check whether the hidden product appears in their top 10. All rows use the same
regenerated data.

| Shoppers | Popularity | Item-item only (original model) | Adaptive Hybrid |
|---|---|---|---|
| Established (full history, 973 shoppers) | 9.0% | 39.8% | **42.9%** |
| New (only their first 3 products kept) | — | 10.4% | **24.6%** |

NDCG@10 (which also rewards ranking the hit higher) for established shoppers: 0.252 → **0.280**.
Catalog coverage is 99% for both personalised models, against 3% for popularity.

## 4. Decisions and trade-offs

| Decision | Why |
|---|---|
| **Keep other sizes of products the shopper already has** | Tried hiding them ("don't show Milo Value Pack after Milo Small Pack"). 46% of hidden next purchases were another size of something already seen, and hiding them dropped the hit rate from 43% to 13%. Rejected. |
| **Adaptive rather than fixed blend** | Content similarity helps most when history is short (10.4% → 24.6%). With long histories, collaborative signal is richer, so its share grows. |
| **Content never drops below 20%** | A fixed 0.2 content share still beat collaborative-only for established shoppers (42.7% vs 39.8%). |
| **Recommendations still update only on Retrain** | Keeps the original teaching flow ("events are pending until retrain"). Content similarity *could* react instantly; that would be a separate change. |
| **Per-product shopper counts in explanations** | Accurate and checkable. "Same history" counts would almost always be 0. |
| **One model name** | Previously there were three names (class, metadata, API). Now `AdaptiveHybridRecommender` / `adaptive_hybrid` throughout. |

**Unchanged from the original:** event weights (view 1, cart 3, purchase 6), popularity fallback
for unknown shoppers, the holdout protocol, the feedback log and `new_interactions.csv` flow, and
the deploy layout.

## 5. Known limitations

- **The shoppers are synthetic.** The generator was built to cluster by category, which naturally favours
  content similarity. The numbers show the approach works on realistic-looking data. Real
  behaviour data is the true test.
- **A brand-new shopper sees popular items until a retrain**, even after their first purchase.
- **Explanations can cite a weak link.** If a shopper's only overlap with a product is a single
  view, the reason says so ("you viewed it"), but the count may be small.
- **Product search, shopper search and the model panel are FastAPI-only.** They need routes of their
  own before the full UI runs on the AWS path.
- **Metrics in the original README** (Recall@10 0.123) were measured on the previous synthetic
  catalog and can't be compared directly with the numbers above.

## 6. Tuning knobs (`backend/src/config.py`)

| Setting | Value | Effect |
|---|---|---|
| `CONTENT_SHARE_MAX` | 0.8 | Content share for shoppers with very short histories |
| `CONTENT_SHARE_MIN` | 0.2 | Content share for long histories |
| `CONTENT_SHARE_HISTORY_SIZE` | 20 | How fast the share falls: `1 − products_seen / 20` |
| `SAME_SUBCATEGORY_BONUS` | 0.5 | How strongly the same subcategory counts as "similar" |
| `SAME_CATEGORY_BONUS` | 0.2 | Same category |
| `SAME_BRAND_BONUS` | 0.1 | Same brand |

## 7. How to reproduce

From `project2_shopsmart/backend/`:

```bash
python scripts/generate_retail_data.py   # only needed when products.csv changes
python -m src.train                      # trains + prints evaluation (hybrid / item-item / popularity)
python -m pytest -q                      # 20 tests
```

To try it in the UI, choose **New guest**, use **Find a product** to buy a Nike sneaker and a
Milo, then press **Retrain model**. The cards mix both categories, and each **Why?** tooltip
names the product behind it.

To undo the data change, copy `backend/data/archive/users.csv` and `interactions.csv` back into
`backend/data/` and retrain. This only makes sense with the old `products.csv`.

## 8. Files changed

| Area | Files |
|---|---|
| Data | `backend/scripts/generate_retail_data.py` (new), `backend/data/users.csv`, `backend/data/interactions.csv`, `backend/data/archive/` (new) |
| Model | `backend/src/recommender.py`, `backend/src/config.py`, `backend/src/train.py`, `backend/src/evaluate.py` |
| API | `backend/api/main.py` |
| AWS | `backend/aws/sagemaker/train.py`, `backend/aws/sagemaker/inference.py`, `backend/aws/sagemaker/package_model.py` (new) |
| UI | `frontend/app/components/RecommendationGrid.tsx`, `frontend/app/components/ProductSearch.tsx` (new), `frontend/app/components/ModelPanel.tsx`, `frontend/app/page.tsx`, `frontend/app/lib/api.ts`, `frontend/app/globals.css` |
| Tests | `backend/tests/test_recommender.py`, `backend/tests/test_api.py` |
| Docs | `README.md`, `docs/hybrid-model-changes.md` (this file) |
