"""Candidate generation: decide which offers are eligible before ranking."""
from datetime import date
from typing import Optional

import pandas as pd

from .config import CANDIDATE_LIMIT
from .features import channel_matches_device


def generate_candidates(
    offers: pd.DataFrame,
    *,
    region: Optional[str] = None,
    device_type: Optional[str] = None,
    today: Optional[date] = None,
    exclude_offer_ids=(),
    enforce_region: bool = True,
    enforce_channel: bool = True,
    offer_popularity: Optional[pd.Series] = None,
    limit: int = CANDIDATE_LIMIT,
):
    """Apply business rules in order and report how many offers each rule removed."""
    # Each rule narrows the pool and records a funnel step, so the UI can show what removed offers.
    candidates = offers[offers["active"]]
    funnel = [{"step": "active offers", "remaining": int(len(candidates))}]

    if today is not None:
        candidates = candidates[candidates["expires_at"] >= today]
        funnel.append({"step": "not expired", "remaining": int(len(candidates))})

    if enforce_region and region is not None:
        candidates = candidates[candidates["region"].isin([region, "all"])]
        funnel.append({"step": f"available in {region}", "remaining": int(len(candidates))})

    # Same channel/device rule the ranker also uses as its channel_match feature.
    if enforce_channel and device_type is not None:
        candidates = candidates[
            channel_matches_device(candidates["channel"], pd.Series(device_type, index=candidates.index)).astype(bool)
        ]
        funnel.append({"step": f"works on {device_type}", "remaining": int(len(candidates))})

    # Offers this user has already redeemed are never recommended again.
    exclude_offer_ids = set(exclude_offer_ids)
    if exclude_offer_ids:
        candidates = candidates[~candidates["offer_id"].isin(exclude_offer_ids)]
        funnel.append({"step": "not already redeemed", "remaining": int(len(candidates))})

    if len(candidates) > limit:
        # Keep the most engaged-with offers when the pool is too large to rank.
        if offer_popularity is not None:
            order = candidates["offer_id"].map(offer_popularity).fillna(0).sort_values(ascending=False)
            candidates = candidates.loc[order.index]
        candidates = candidates.head(limit)
        funnel.append({"step": f"top {limit} by popularity", "remaining": int(len(candidates))})

    return candidates, funnel
