"""Policy thresholds: ALLOW/REVIEW/BLOCK boundaries, validation and save/load."""
import pytest

from src.policy import DEFAULT_POLICY, Policy, decide, load_policy, save_policy


def test_low_score_allows():
    assert decide(0.20) == "ALLOW"


def test_middle_score_reviews():
    assert decide(0.70) == "REVIEW"


def test_high_score_blocks():
    assert decide(0.95) == "BLOCK"


@pytest.mark.parametrize("score, expected", [
    (DEFAULT_POLICY.review_threshold - 1e-9, "ALLOW"),
    (DEFAULT_POLICY.review_threshold, "REVIEW"),
    (DEFAULT_POLICY.block_threshold - 1e-9, "REVIEW"),
    (DEFAULT_POLICY.block_threshold, "BLOCK"),
    (0.0, "ALLOW"),
    (1.0, "BLOCK"),
])
def test_threshold_boundaries(score, expected):
    assert decide(score) == expected


def test_policy_change_needs_no_model():
    stricter = Policy(review_threshold=0.3, block_threshold=0.6, version="policy-test")
    assert stricter.decide(0.4) == "REVIEW" and DEFAULT_POLICY.decide(0.4) == "ALLOW"


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        Policy(review_threshold=0.9, block_threshold=0.5)
    with pytest.raises(ValueError):
        Policy(review_threshold=-0.1, block_threshold=0.5)


def test_policy_round_trip(tmp_path):
    path = tmp_path / "policy.json"
    save_policy(Policy(0.4, 0.7, "policy-v9"), path)
    assert load_policy(path) == Policy(0.4, 0.7, "policy-v9")
    assert load_policy(tmp_path / "missing.json") == DEFAULT_POLICY
