"""Central configuration: file paths, implicit-feedback event weights and the default Top-K.

Every other module imports from here, so changing a weight or path here changes the whole pipeline.
"""
from pathlib import Path

# Backend root (project2_shopsmart/backend/), resolved from this file's location so it works from any cwd.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# Behaviour captured after the original dataset (from the UI or experiments).
# Kept in its own file so the supplied CSVs stay untouched.
NEW_INTERACTIONS_PATH = DATA_DIR / "new_interactions.csv"

# Recommendation feedback (impressions, clicks, ...) is logged separately from
# business transactions, as section 22 of the guide recommends.
FEEDBACK_LOG_PATH = ROOT / "feedback" / "events.jsonl"

# Teaching defaults, not universal truths. Treat them as tunable assumptions.
EVENT_WEIGHTS = {
    "view": 1.0,
    "cart": 3.0,
    "purchase": 6.0,
}

# Events strong enough to count as a "positive" when evaluating.
POSITIVE_EVENT_TYPES = {"cart", "purchase"}

DEFAULT_TOP_K = 10

# Hybrid blend: share of each recommendation score that comes from content similarity
# ("similar products") rather than collaborative filtering ("shoppers like you").
# share = clip(1 - distinct_items_seen / CONTENT_SHARE_HISTORY_SIZE, MIN, MAX), so a shopper with
# 3 products gets 0.8 (mostly similar products) and one with 16+ products gets 0.2.
CONTENT_SHARE_MAX = 0.8
CONTENT_SHARE_MIN = 0.2
CONTENT_SHARE_HISTORY_SIZE = 20

# Content similarity = TF-IDF text similarity plus these bonuses for shared catalog attributes.
SAME_SUBCATEGORY_BONUS = 0.5
SAME_CATEGORY_BONUS = 0.2
SAME_BRAND_BONUS = 0.1
