"""Shared by the API and stream Lambdas: validate → idempotency → model → policy.

Policy thresholds come from environment variables, so the policy can change
without retraining or redeploying the SageMaker model.
"""
import json
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

# Clients are created once per Lambda container (outside the handler) and reused across invocations.
sagemaker_runtime = boto3.client("sagemaker-runtime")
decisions_table = boto3.resource("dynamodb").Table(os.environ["DECISIONS_TABLE"])
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT"]
REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.55"))
BLOCK_THRESHOLD = float(os.environ.get("BLOCK_THRESHOLD", "0.85"))
POLICY_VERSION = os.environ.get("POLICY_VERSION", "policy-v1")

REQUIRED_FIELDS = ["transaction_id", "customer_id", "amount", "hour", "velocity_1h", "country", "device", "merchant_category"]


class InvalidTransaction(ValueError):
    """A malformed record: reject it (422 / skip) rather than retry it."""
    pass


def decide(risk_score: float) -> str:
    """Same ALLOW/REVIEW/BLOCK rule as src.policy.Policy.decide, using the env-var thresholds."""
    if risk_score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if risk_score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "ALLOW"


def validate(transaction: dict) -> dict:
    """Check required fields, forbid client risk_score, check numeric ranges.

    Returns only the whitelisted fields (plus currency if present), dropping anything else.
    """
    missing = [field for field in REQUIRED_FIELDS if transaction.get(field) in (None, "")]
    if missing:
        raise InvalidTransaction(f"Missing fields: {missing}")
    if "risk_score" in transaction:
        raise InvalidTransaction("Clients may not supply risk_score")
    try:
        amount, hour, velocity = float(transaction["amount"]), int(transaction["hour"]), int(transaction["velocity_1h"])
    except (TypeError, ValueError):
        raise InvalidTransaction("amount, hour and velocity_1h must be numeric")
    if amount <= 0 or not 0 <= hour <= 23 or velocity < 1:
        raise InvalidTransaction("amount must be > 0, hour 0-23, velocity_1h >= 1")
    return {field: transaction[field] for field in REQUIRED_FIELDS + ["currency"] if field in transaction}


def score_and_decide(transaction: dict) -> dict:
    """Idempotent: a retried transaction_id returns the stored decision."""
    transaction = validate(transaction)
    # Fast path: this transaction_id was already decided, so return the stored response.
    stored = decisions_table.get_item(Key={"transaction_id": transaction["transaction_id"]}).get("Item")
    if stored:
        return {**json.loads(stored["response"]), "idempotent_replay": True}

    # The SageMaker endpoint returns only a risk score; the policy is applied here.
    endpoint_response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME, ContentType="application/json", Body=json.dumps(transaction).encode("utf-8"),
    )
    prediction = json.loads(endpoint_response["Body"].read().decode("utf-8"))
    result = {
        "transaction_id": transaction["transaction_id"],
        "risk_score": prediction["risk_score"],
        "decision": decide(float(prediction["risk_score"])),
        "model_version": prediction["model_version"],
        "policy_version": POLICY_VERSION,
        "idempotent_replay": False,
    }
    try:
        # Conditional put: the write succeeds only if no item with this transaction_id exists yet,
        # so two concurrent deliveries cannot both store a decision. DynamoDB needs Decimal, not float.
        decisions_table.put_item(
            Item={"transaction_id": result["transaction_id"], "decision": result["decision"],
                  "risk_score": Decimal(str(result["risk_score"])), "response": json.dumps(result),
                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            ConditionExpression="attribute_not_exists(transaction_id)",
        )
    except ClientError as error:
        if error.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
        # A concurrent delivery won the race: return what it stored.
        stored = decisions_table.get_item(Key={"transaction_id": result["transaction_id"]})["Item"]
        return {**json.loads(stored["response"]), "idempotent_replay": True}
    print(json.dumps({"event": "decision", **result}))  # structured CloudWatch log line
    return result
