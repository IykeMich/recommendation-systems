"""Policy layer: turns a risk score into ALLOW / REVIEW / BLOCK.

Model = estimates risk. Policy = decides what to do with it. Thresholds live in
artifacts/policy.json and can change without retraining the model.
"""
import json
from dataclasses import asdict, dataclass

from .config import DEFAULT_BLOCK_THRESHOLD, DEFAULT_REVIEW_THRESHOLD, POLICY_PATH

DECISIONS = ("ALLOW", "REVIEW", "BLOCK")


@dataclass(frozen=True)
class Policy:
    """Immutable ALLOW/REVIEW/BLOCK thresholds plus a version string recorded on each decision."""
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD
    version: str = "policy-v1"

    def __post_init__(self):
        """Reject threshold pairs that are out of [0, 1] or in the wrong order."""
        if not 0 <= self.review_threshold <= self.block_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= review_threshold <= block_threshold <= 1")

    def decide(self, risk_score: float) -> str:
        """BLOCK at or above block_threshold, REVIEW at or above review_threshold, else ALLOW."""
        if risk_score >= self.block_threshold:
            return "BLOCK"
        if risk_score >= self.review_threshold:
            return "REVIEW"
        return "ALLOW"


DEFAULT_POLICY = Policy()


def decide(risk_score: float, policy: Policy = DEFAULT_POLICY) -> str:
    """Module-level shortcut: apply `policy` (default thresholds if omitted) to one score."""
    return policy.decide(risk_score)


def load_policy(path=POLICY_PATH) -> Policy:
    """Read thresholds from policy.json, or fall back to the default policy if the file is absent."""
    if not path.exists():
        return DEFAULT_POLICY
    return Policy(**json.loads(path.read_text()))


def save_policy(policy: Policy, path=POLICY_PATH) -> None:
    """Write the policy to JSON (creating the folder if needed) so it survives restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(policy), indent=2))
