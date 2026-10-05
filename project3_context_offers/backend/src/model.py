"""The ranker: an sklearn logistic-regression pipeline, plus helpers that read its coefficients
and turn per-feature contributions into human-readable reasons for each recommended offer."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(numeric_features, categorical_features):
    """Numeric: median-impute then standardise (0 = training average). Categorical: impute the most
    frequent value then one-hot; categories unseen in training are ignored at serving time."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Learned weight per (transformed) feature, largest effect first (experiment 8)."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def feature_contributions(model: Pipeline, features: pd.DataFrame):
    """Per-row contribution of each feature to the logit (coefficient x transformed value),
    plus the transformed values themselves (above/below average for numeric features)."""
    transformed = model.named_steps["preprocessor"].transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    # Logit = intercept + sum(coef_i * x_i), so coef_i * x_i is exactly feature i's share of the score.
    contributions = transformed * model.named_steps["model"].coef_[0]
    return (
        pd.DataFrame(contributions, columns=feature_names, index=features.index),
        pd.DataFrame(transformed, columns=feature_names, index=features.index),
    )


# (label when the value is above average, label when below average)
NUMERIC_REASONS = {
    "category_affinity": ("Engaged with this category before", "New category for them"),
    "preferred_category_match": ("Matches their preferred category", "Outside their preferred category"),
    "offer_popularity": ("Popular with other shoppers", "Less crowded offer"),
    "positive_rate": ("Often engages with offers", "Rarely engages with offers"),
    "positive_count": ("Active offer user", "Few past engagements"),
    "interaction_count": ("Long offer history", "Short offer history"),
    "discount_pct": ("Big discount", "Smaller discount"),
    "min_spend": ("Higher minimum spend", "Low minimum spend"),
    "region_match": ("Available in their region", "Outside their region"),
    "channel_match": ("Works on their device", "Not built for their device"),
    "is_weekend": ("Weekend timing", "Weekday timing"),
    "hour_sin": ("Time of day", "Time of day"),
    "hour_cos": ("Time of day", "Time of day"),
    "dow_sin": ("Day of week", "Day of week"),
    "dow_cos": ("Day of week", "Day of week"),
}

CHANNEL_NAMES = {"both": "Web & app", "web": "Web-only", "mobile": "App-only"}


def describe_feature(transformed_name: str, is_above_average: bool = True) -> str:
    """Human-readable label for a transformed feature name like cat__category_x_daypart_Home|evening."""
    # sklearn prefixes names with the transformer: 'num__discount_pct' or 'cat__region_Lagos'.
    kind, _, name = transformed_name.partition("__")
    if kind == "num":
        high_label, low_label = NUMERIC_REASONS.get(name, (name, name))
        return high_label if is_above_average else low_label
    for prefix, template in [
        ("category_x_daypart_", "{0} offers in the {1}"),
        ("category_x_device_", "{0} offers on {1}"),
    ]:
        if name.startswith(prefix):
            return template.format(*name[len(prefix):].split("|"))
    if name.startswith("channel_"):
        channel = name[len("channel_"):]
        return f"{CHANNEL_NAMES.get(channel, channel)} offer"
    for prefix, template in [
        ("segment_", "{0} segment"),
        ("category_", "{0} offer"),
        ("device_type_", "On {0}"),
        ("region_", "In {0}"),
    ]:
        if name.startswith(prefix):
            return template.format(name[len(prefix):])
    return name


def top_reasons(contribution_row: pd.Series, value_row: pd.Series, count: int = 3) -> list:
    """The features that pushed this offer's score up the most."""
    # Only features that raised the score count as reasons, strongest first.
    positive = contribution_row[contribution_row > 0].sort_values(ascending=False)
    reasons = []
    for transformed_name, contribution in positive.items():
        # A scaled value >= 0 means at or above the training average, which picks the 'high' label.
        # Several features share a label (hour_sin/hour_cos), so duplicates are skipped.
        label = describe_feature(transformed_name, value_row[transformed_name] >= 0)
        if label not in {reason["label"] for reason in reasons}:
            reasons.append({"label": label, "weight": round(float(contribution), 3)})
        if len(reasons) == count:
            break
    return reasons
