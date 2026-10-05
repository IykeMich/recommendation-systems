"""The ranker: a scikit-learn logistic-regression pipeline, plus helpers to read its coefficients.

It only scores products that already passed the eligibility rules.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ranker(numeric_features, categorical_features):
    """Preprocessing + logistic regression in one Pipeline, so serving reuses the fitted transforms.

    Numeric: median-impute then standardise. Categorical: mode-impute then one-hot
    (unknown categories at serving time are ignored instead of raising).
    """
    preprocessor = ColumnTransformer([
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
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        # Strong regularisation: the behavioural signal in this data is weak.
        ("model", LogisticRegression(max_iter=3000, C=0.1)),
    ])


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Transformed feature names with their coefficients, largest absolute effect first."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# Human-readable labels for the numeric features shown in the UI's model panel.
NUMERIC_LABELS = {
    "previously_engaged": "Engaged with this product before",
    "prior_product_events": "Number of past events on this product",
    "category_affinity": "Engaged with this category before",
    "product_popularity": "Product popularity",
    "income_headroom": "Income well above the minimum",
    "existing_products": "Number of existing products",
}


def describe_feature(transformed_name: str) -> str:
    """Turn a transformed name such as "cat__risk_fit_low|high" into readable text for the UI."""
    # ColumnTransformer prefixes names with "num__" or "cat__"; one-hot adds "<column>_<value>".
    kind, _, name = transformed_name.partition("__")
    if kind == "num":
        return NUMERIC_LABELS.get(name, name)
    # Crossed features first: their value holds two parts joined by "|".
    for prefix, template in [
        ("risk_fit_", "{0} risk profile × {1} risk product"),
        ("employment_x_category_", "{0} × {1}"),
    ]:
        if name.startswith(prefix):
            return template.format(*name[len(prefix):].replace("_", " ").split("|"))
    # Then plain one-hot categories.
    for prefix, template in [
        ("category_", "{0} product"),
        ("risk_level_", "{0}-risk product"),
        ("age_band_", "Age {0}"),
    ]:
        if name.startswith(prefix):
            return template.format(name[len(prefix):])
    return name
