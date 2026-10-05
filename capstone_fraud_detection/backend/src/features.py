"""Feature engineering.

Golden rule: a transaction's features may only use information available when
it is scored. This dataset has no timestamps, so "history" means the
customer's transactions in the TRAINING split only:
  * validation/test/serving rows see profiles built from training rows;
  * training rows see their own customer's profile with THEMSELVES left out
    (leave-one-out), so no row describes itself.
"""
import numpy as np
import pandas as pd

# Model input columns. hour_sin/hour_cos encode hour on a circle so 23:00 and 00:00 are close.
NUMERIC = [
    "amount",
    "amount_log1p",
    "hour",
    "hour_sin",
    "hour_cos",
    "velocity_1h",
    "customer_txn_count",
    "amount_vs_customer_mean",
    "customer_device_share",
    "customer_country_share",
]

CATEGORICAL = [
    "country",
    "device",
    "merchant_category",
]

# Feature groups used by the ablation experiments.
VELOCITY_FEATURES = ["velocity_1h"]
NOVELTY_FEATURES = ["customer_device_share", "customer_country_share"]


class CustomerProfiles:
    """Per-customer reference history: count, amount total, device and country counts."""

    def fit(self, history: pd.DataFrame) -> "CustomerProfiles":
        """Count each customer's transactions, amount total, and per-device / per-country usage.

        Only ever fitted on the TRAINING split (see module docstring).
        """
        grouped = history.groupby("customer_id")
        self.txn_count = grouped.size()
        self.amount_sum = grouped["amount"].sum()
        self.device_counts = history.groupby(["customer_id", "device"]).size()
        self.country_counts = history.groupby(["customer_id", "country"]).size()
        return self

    def features_for(self, rows: pd.DataFrame, leave_one_out: bool = False) -> pd.DataFrame:
        """Customer-history features for `rows`, looked up from the fitted profiles.

        With leave_one_out=True (training rows only) each row's own transaction is subtracted
        from its customer's counts and amount total, so the feature never contains the row itself.
        """
        # Leave-one-out: remove this row's own contribution (1 transaction, its amount, its
        # device and country) from the totals. Unknown customers map to NaN, filled with 0.
        own = 1 if leave_one_out else 0
        count = rows["customer_id"].map(self.txn_count).fillna(0).to_numpy() - own
        amount_sum = rows["customer_id"].map(self.amount_sum).fillna(0).to_numpy() - (rows["amount"].to_numpy() if leave_one_out else 0)
        device_count = self._pair_counts(self.device_counts, rows, "device") - own
        country_count = self._pair_counts(self.country_counts, rows, "country") - own

        # Customers with no (other) history get NaN for mean/ratio instead of dividing by zero;
        # the pipeline's imputer fills those NaNs later. Shares fall back to 0 ("never seen").
        safe_count = np.where(count > 0, count, np.nan)
        customer_mean = amount_sum / safe_count
        return pd.DataFrame({
            "customer_txn_count": count,
            "customer_mean_amount": customer_mean,
            "amount_vs_customer_mean": rows["amount"].to_numpy() / customer_mean,
            "customer_device_share": np.nan_to_num(device_count / safe_count, nan=0.0),
            "customer_country_share": np.nan_to_num(country_count / safe_count, nan=0.0),
        }, index=rows.index)

    @staticmethod
    def _pair_counts(counts: pd.Series, rows: pd.DataFrame, column: str) -> np.ndarray:
        """Look up the (customer_id, value) count for each row; unseen pairs give 0.

        The counts Series has a (customer, device|country) MultiIndex, so reindexing it with one
        key per row returns the counts aligned row-for-row with `rows`.
        """
        keys = pd.MultiIndex.from_arrays([rows["customer_id"], rows[column]])
        return counts.reindex(keys).fillna(0).to_numpy()

    def profile(self, customer_id: str) -> dict:
        """Readable profile for the dashboard's alert drawer."""
        count = int(self.txn_count.get(customer_id, 0))
        if count == 0:
            return {"customer_id": customer_id, "known": False}
        return {
            "customer_id": customer_id,
            "known": True,
            "reference_transactions": count,
            "mean_amount": round(float(self.amount_sum[customer_id] / count), 2),
            "devices": {device: int(n) for device, n in self.device_counts[customer_id].items()},
            "countries": {country: int(n) for country, n in self.country_counts[customer_id].items()},
        }


def build_features(rows: pd.DataFrame, profiles: CustomerProfiles, leave_one_out: bool = False) -> pd.DataFrame:
    """Add amount/hour transforms and the customer-profile features to a copy of `rows`."""
    features = rows.copy()
    # log1p compresses the long tail of amounts; clip guards against negative values.
    features["amount_log1p"] = np.log1p(features["amount"].clip(lower=0))
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    return features.join(profiles.features_for(features, leave_one_out=leave_one_out))
