"""API Gateway -> Lambda -> Kinesis (guide sections 27 and 32).

Route: POST /v1/offers/events
Stores the context that produced the recommendation with every event, so later
analysis knows the conditions under which an offer was clicked or redeemed.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import boto3

kinesis = boto3.client("kinesis")
STREAM_NAME = os.environ["FEEDBACK_STREAM_NAME"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
ALLOWED_EVENT_TYPES = {"recommendation_impression", "recommendation_click", "click", "redeem"}


def response(status_code, body):
    """API Gateway proxy response with a JSON body and a CORS header."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": ALLOWED_ORIGIN},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """Validate a feedback event, fill in event_id/timestamp if missing, and put it on Kinesis."""
    feedback_event = json.loads(event.get("body") or "{}")

    missing_fields = [field for field in ("user_id", "offer_id", "event_type", "context") if not feedback_event.get(field)]
    if missing_fields:
        return response(400, {"detail": f"Missing fields: {missing_fields}"})
    if feedback_event["event_type"] not in ALLOWED_EVENT_TYPES:
        return response(400, {"detail": f"Unknown event_type: {feedback_event['event_type']}"})

    feedback_event.setdefault("event_id", f"evt-{uuid.uuid4().hex[:12]}")
    feedback_event.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Partitioning by user_id keeps each user's events in order within one shard.
    # 202 Accepted: the event is queued for downstream processing, not yet applied.
    kinesis.put_record(
        StreamName=STREAM_NAME,
        Data=json.dumps(feedback_event).encode("utf-8"),
        PartitionKey=feedback_event["user_id"],
    )
    return response(202, {"accepted": True, "event_id": feedback_event["event_id"]})
