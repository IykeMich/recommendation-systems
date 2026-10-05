# Command-line tests and experiments

How to run ShopSmart's command-line checks, read their output, and write new ones. The worked
example covers the guide's new-item (cold-start) brief:

> Test two cases separately:
> • Brand-new product with zero interactions: a pure collaborative model should not have evidence to recommend it.
> • New product that receives views/carts/purchases: rebuild the model and check whether it becomes a candidate.

All commands run from `project2_shopsmart/backend/` with the virtual environment active:

```bash
cd project2_shopsmart/backend
source ../.venv/bin/activate        # or: source .venv/bin/activate if you created it in backend/
```

## 1. Two kinds of command-line checks

| | Automated tests | Experiments |
|---|---|---|
| Command | `python -m pytest -q` | `python -m src.experiments` |
| Lives in | `tests/test_recommender.py`, `tests/test_api.py` | `src/experiments.py` (also called from the notebook) |
| Answers | "Is it still correct?" Pass/fail, no reading needed | "How does it behave?" Prints numbers and rankings for a human to interpret |
| Data | Small toy tables (model tests) or the real data in a temp folder (API tests) | The real CSVs, changed **in memory only** |
| Changes files? | No. API tests redirect every write to a temp folder | No |

Use an **experiment** to explore and explain a behaviour. Once you know what *should* happen,
lock it in with a **test** (§5) so a future change can't silently break it.

## 2. Running them

```bash
python -m pytest -q                                  # all tests (about 20 s)
python -m pytest -q tests/test_recommender.py        # model tests only (fast, toy data)
python -m pytest -q -k "category"                    # tests whose name contains "category"
python -m pytest -v                                  # one line per test, with names

python -m src.experiments                            # both built-in experiments
python -m src.train                                  # training + offline evaluation (prints metrics)
```

A failing test prints the assertion that failed and the values involved. Fix the code, or if the
behaviour changed on purpose, fix the test, and say why in the docs.

## 3. The built-in experiments and how to read them

`python -m src.experiments` runs two experiments. Both use the **collaborative-only**
`ItemItemRecommender`, which is the model the guide starts with.

**New interaction experiment** (guide §14): shopper `R00001` buys the most popular product, the model is
rebuilt, and their top 10 is printed before and after:

```
=== New interaction experiment for R00001 ===
Adding a purchase of P00001 (Food & Beverage)
Top-10 after the new purchase (movement vs before):
   1. P00326 Colgate Optic White - Value Pack score=   2.56  (+0)
   5. P00295 L'Oréal Revitalift Cream - Small Pack score=   2.45  (+2)
   6. P00463 Dyson V15 Detect - Standard  score=   2.42  (-1)
  10. P00469 Philips Sonicare Toothbrush - Small Pack score=   2.19  (new)
P00001 itself is now 'seen', so it is filtered out: True
```

(Excerpt.) `(+n)` / `(-n)` / `(new)` show rank movement compared with before the purchase. The last line checks that bought products aren't recommended back.

**New item experiment** (guide §15): the cold-start brief, collaborative-only:

```
Zero interactions: P_NEW known to the model? False
After 37 P00137 buyers purchase P_NEW: its rank in R00025's top-10 = 1
```

- Case 1: a product nobody has touched isn't even a column in the model, so it can't be recommended.
- Case 2: once many buyers of an anchor product also buy it, it jumps to #1 for a held-out buyer of that anchor.

What it doesn't show: what happens with **less** evidence, with **views and carts** rather than only
purchases, and how our **Adaptive Hybrid** handles case 1. The worked example below adds all three.

## 4. Writing a new experiment (worked example)

### The recipe

Every experiment follows the same five steps:

1. **Load** the real data: `_, products, events = load_data()`.
2. **Change a copy in memory.** Add rows with `make_event(user, item, event_type)` and `pd.concat`.
   Never write to the CSVs.
3. **Rebuild** the model(s) on the changed copy: `build_user_item(prepare_events(events))`, then `.fit(...)`.
4. **Measure** one thing, such as a rank, a score or whether something is present.
5. **Print** a short line per case that a reader can follow without the code.

Pick test shoppers **deterministically** (for example, "the shopper with the most snack events"), so
every run prints the same thing.

### The code

Add this to `src/experiments.py` (the imports go at the top of the file):

```python
from .recommender import AdaptiveHybridRecommender, ItemItemRecommender

# A realistic new product: real attributes, zero interactions.
NEW_PRODUCT = {
    "item_id": "P_NEW", "name": "Crunchy Plantain Chips - Small Pack",
    "category": "Food & Beverage", "subcategory": "Snacks", "brand": "Naija Crunch",
    "price": 2.0, "description": "Salted plantain chips for snacking.", "stock": 100,
}


def rank_of(item_id, recommendations):
    """1-based position of item_id in a recommendation list, or None if absent."""
    return next((rank for rank, r in enumerate(recommendations, 1) if r["item_id"] == item_id), None)


def new_item_hybrid_experiment(events: pd.DataFrame, products: pd.DataFrame):
    """Cold start for a new product: collaborative-only vs Adaptive Hybrid, with growing evidence."""
    print("\n=== New item experiment: collaborative vs adaptive hybrid ===")
    new_id = NEW_PRODUCT["item_id"]
    catalog = pd.concat([products, pd.DataFrame([NEW_PRODUCT])], ignore_index=True)

    def fit_both(event_rows):
        matrix = build_user_item(prepare_events(event_rows))
        return ItemItemRecommender().fit(matrix), AdaptiveHybridRecommender(catalog).fit(matrix)

    # Snack shoppers, most active first: the first is the established test shopper,
    # the rest create evidence for the new product in case 2.
    snack_ids = set(products.loc[products.subcategory == "Snacks", "item_id"])
    snack_fans = events[events.item_id.isin(snack_ids)].user_id.value_counts().index.tolist()
    regular, evidence_shoppers = snack_fans[0], snack_fans[1:]

    # A brand-new snack shopper: three snack purchases, nothing else.
    first_snacks = products.loc[products.subcategory == "Snacks", "item_id"].tolist()[:3]
    newcomer_events = pd.DataFrame([make_event("SNACK_NEWBIE", item_id, "purchase") for item_id in first_snacks])
    events = pd.concat([events, newcomer_events], ignore_index=True)

    # Case 1: zero interactions.
    collaborative, hybrid = fit_both(events)
    print(f"Case 1 - zero interactions. In the collaborative model? {new_id in set(collaborative.item_ids)}")
    print(f"  new snack shopper:         collaborative rank {rank_of(new_id, collaborative.recommend('SNACK_NEWBIE', k=10))}, "
          f"hybrid rank {rank_of(new_id, hybrid.recommend('SNACK_NEWBIE', k=10))}")
    print(f"  established snack shopper: collaborative rank {rank_of(new_id, collaborative.recommend(regular, k=10))}, "
          f"hybrid rank {rank_of(new_id, hybrid.recommend(regular, k=10))} "
          f"(full list: {rank_of(new_id, hybrid.recommend(regular, k=len(catalog)))})")

    # Case 2: growing evidence. Shopper i views it; every 2nd also carts; every 3rd also buys.
    print(f"Case 2 - evidence grows (ranks in {regular}'s top 10, rebuilt each time)")
    for shopper_count in [1, 3, 5, 10]:
        new_rows = [
            make_event(shopper, new_id, event_type)
            for index, shopper in enumerate(evidence_shoppers[:shopper_count])
            for event_type in ["view", "cart", "purchase"][: index % 3 + 1]
        ]
        collaborative, hybrid = fit_both(pd.concat([events, pd.DataFrame(new_rows)], ignore_index=True))
        print(f"  {shopper_count:>2} shoppers ({len(new_rows):>2} events): "
              f"collaborative {rank_of(new_id, collaborative.recommend(regular, k=10))}, "
              f"hybrid {rank_of(new_id, hybrid.recommend(regular, k=10))}")
```

Then call it from the `__main__` block at the bottom:

```python
    new_item_hybrid_experiment(interaction_events, catalog)
```

### The output (current data)

```
=== New item experiment: collaborative vs adaptive hybrid ===
Case 1 - zero interactions. In the collaborative model? False
  new snack shopper:         collaborative rank None, hybrid rank 5
  established snack shopper: collaborative rank None, hybrid rank None (full list: 164)
Case 2 - evidence grows (ranks in R00784's top 10, rebuilt each time)
   1 shoppers ( 1 events): collaborative None, hybrid 7
   3 shoppers ( 6 events): collaborative 2, hybrid 2
   5 shoppers ( 9 events): collaborative 1, hybrid 2
  10 shoppers (19 events): collaborative 2, hybrid 2
```

### How to read it

| Result | Meaning |
|---|---|
| Collaborative: `None` in case 1 | The brief's first point. With no interactions, the product isn't in the model at all, so there is no evidence. |
| Hybrid #5 for the **new** snack shopper | Content similarity (same subcategory, "Snacks") finds the product with zero interactions. The shopper's short history gives content an 80% share. This is why production systems add content features. |
| Hybrid #164 for the **established** shopper | Their long history gives content only 20%, and the product has no collaborative evidence yet, so it sinks. The hybrid can *reach* it but doesn't *push* it. Pushing a new product is the job of business rules ("new arrivals" slots) and exploration (deliberately showing a little untested stock), the other two items in the brief's lesson. |
| Case 2: 1 shopper → collaborative still `None`, hybrid #7 | One view from one shopper is too little overlap for collaborative filtering. The hybrid combines that small signal with content and gets it into the top 10. |
| 3+ shoppers → both at #1–2 | The brief's second point. Once a handful of overlapping shoppers interact, the collaborative model treats it as a strong candidate. |

The exact numbers depend on the data. They will shift if you regenerate `interactions.csv` or
change `products.csv`, but the pattern should hold.

## 5. Turning an experiment into a test

When an experiment shows behaviour you want to guarantee, assert it in `tests/`. Use small toy data
so the test is fast and the expected answer is obvious. This one locks in case 1 for both models:

```python
def test_zero_interaction_product_needs_content(toy_events, toy_products):
    """E has no interactions: collaborative-only can't see it, the hybrid reaches it through content."""
    matrix = build_user_item(prepare_events(toy_events))
    assert "E" not in set(ItemItemRecommender().fit(matrix).item_ids)
    hybrid_ids = [r["item_id"] for r in AdaptiveHybridRecommender(toy_products).fit(matrix).recommend("u1", k=5)]
    assert "E" in hybrid_ids
```

`toy_events` and `toy_products` are fixtures already defined in `tests/test_recommender.py`. A
similar test, `test_hybrid_reaches_never_interacted_items_and_explains`, already exists there.

Rules of thumb for tests:
- **One behaviour per test**, named after the behaviour (`test_new_shopper_buying_one_category_gets_that_category`).
- **Assert outcomes, not exact scores.** "In the top 10" or "at least 70% Food & Beverage" survive harmless tuning; `score == 0.6171` doesn't.
- **Never write to `data/`, `artifacts/` or `feedback/`.** API tests use the `client` fixture, which redirects every write to a temporary folder.
- **Run the whole suite before and after a change**, so you know any failure is yours.

## 6. Checklist for a new command-line check

- [ ] It runs from `backend/` with one command.
- [ ] It changes data in memory only.
- [ ] Test shoppers and products are chosen deterministically.
- [ ] The output has one short line per case and needs no code to understand.
- [ ] Any behaviour it shows that must not regress is also a pytest test.
- [ ] `README.md` and this document mention it (project docs are updated with every change).
