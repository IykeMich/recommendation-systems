"""Central configuration: paths, CSV-to-internal schema adapters, label rule and constants.

Every other module (data loading, eligibility, features, training, API) imports from here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# Feedback captured after the original dataset; the supplied CSVs stay untouched.
NEW_INTERACTIONS_PATH = DATA_DIR / "new_interactions.csv"
FEEDBACK_LOG_PATH = ROOT / "feedback" / "events.jsonl"

# With 20 products (about 18 eligible per user), a top 10 would be half the catalog.
TOP_K = 5

# Adapters from raw CSV names to the internal schema (guide cell 4).
USER_MAP = {
    "user_id": "user_id",
    "age_band": "age_band",
    "monthly_income": "monthly_income",
    "employment": "employment",
    "risk_profile": "risk_profile",
    "region": "region",
    "existing_products": "existing_products",
}
PRODUCT_MAP = {
    "product_id": "product_id",
    "name": "name",
    "category": "category",
    "risk_level": "risk_level",
    "min_income": "min_income",
    "description": "description",
}
INTERACTION_MAP = {
    "user_id": "user_id",
    "product_id": "product_id",
    "event_type": "event_type",
    "timestamp": "timestamp",
}

EVENT_TYPES = ["view", "learn", "apply"]
# The outcome the ranker predicts: the user applied for the product.
POSITIVE_EVENT_TYPES = {"apply"}

# Ordinal scale so a product's risk level can be compared with a user's risk profile.
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

# Training negatives: eligible products sampled for the same user and moment.
NEGATIVES_PER_POSITIVE = 8
RANDOM_SEED = 42
TEST_FRACTION = 0.2

# Allowed values for the what-if profile editor (served by /v1/profile/options).
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
EMPLOYMENT_TYPES = ["salary", "self_employed", "business_owner", "student"]
RISK_PROFILES = ["low", "medium", "high"]
