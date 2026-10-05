"""Candidate generation: keep only products that pass the hard rules (and are available).

Sits between the eligibility layer and the ranker: the model only ever sees these candidates.
"""
import pandas as pd

from .eligibility import eligibility_report


def filter_available(candidates: pd.DataFrame, *, active_only: bool = True) -> pd.DataFrame:
    """Deterministic availability constraints live outside the model too.
    The supplied catalog has no `active` column, so this is a no-op for it."""
    if active_only and "active" in candidates.columns:
        return candidates[candidates["active"].astype(bool)]
    return candidates


def generate_eligible_candidates(user: dict, products: pd.DataFrame):
    """Returns (eligible candidates, full per-product eligibility report)."""
    report = eligibility_report(user, products)
    candidates = filter_available(report[report["eligible"]])
    return candidates, report
