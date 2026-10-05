"""API Gateway (HTTP API, Lambda proxy) → Lambda → SageMaker → policy (guide section 21).

Route: POST /score
"""
import json
import os

from fraud_common import InvalidTransaction, score_and_decide

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")


def response(status_code: int, body: dict) -> dict:
    """Build an API Gateway proxy response with JSON body and CORS header."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": ALLOWED_ORIGIN},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """POST /score: parse the JSON body, score it, and map validation errors to 422.

    Other errors (e.g. SageMaker failures) propagate, so API Gateway returns a 5xx.
    """
    try:
        transaction = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return response(400, {"detail": "Body must be JSON"})
    try:
        return response(200, score_and_decide(transaction))
    except InvalidTransaction as error:
        print(json.dumps({"event": "rejected", "reason": str(error)}))
        return response(422, {"detail": str(error)})
