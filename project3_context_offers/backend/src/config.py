"""Central settings for Project 3: paths, the raw-to-internal column map, the label rule,
negative-sampling and time-split settings, and the context values the API offers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

# Feedback captured after the original dataset. The supplied CSVs stay untouched.
NEW_INTERACTIONS_PATH = DATA_DIR / "new_offer_interactions.csv"
FEEDBACK_LOG_PATH = ROOT / "feedback" / "events.jsonl"

# Offers returned per request, and the most candidates passed on to the ranker.
TOP_K = 10
CANDIDATE_LIMIT = 100

# Adapter from raw CSV column names to the internal schema (guide cell 3).
COLUMN_MAP = {
    "user_id": "user_id",
    "offer_id": "offer_id",
    "timestamp": "timestamp",
    "event_type": "event_type",
    "device_type": "device_type",
    "region": "region",
}

# The dataset has no binary label, so the rule is explicit: an offer the user
# clicked or redeemed is a positive. Impressions alone are not treated as
# "rejections" because the user may simply not have noticed the offer.
POSITIVE_EVENT_TYPES = {"click", "redeem"}
EVENT_TYPES = {"impression", "click", "redeem"}

# Training negatives: offers sampled from the catalog for the same request
# context that the user did not engage with.
NEGATIVES_PER_POSITIVE = 4
RANDOM_SEED = 42

# Time-aware holdout: the latest 20% of events are the test period.
TEST_FRACTION = 0.2

# Context values the API exposes to the UI (GET /v1/context/options).
DEVICE_TYPES = ["mobile", "tablet", "web"]
REGIONS = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Enugu"]
