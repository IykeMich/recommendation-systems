"""API Gateway -> Lambda -> Kinesis Data Stream (section 22).

Route: POST /v1/events
Kinesis Firehose (or a consumer Lambda) then lands the stream in S3, where
Glue catalogs it and Athena queries it for analysis and future training.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import boto3

# Created once per Lambda container (outside the handler) so warm invocations reuse the client.
kinesis = boto3.client("kinesis")
STREAM_NAME = os.environ["FEEDBACK_STREAM_NAME"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

ALLOWED_EVENT_TYPES = {
    "recommendation_impression",
    "recommendation_click",
    "view",
    "cart",
    "purchase",
}


def response(status_code, body):
    """Build an API Gateway proxy response with JSON body and CORS header."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        },
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """Validate a feedback event from the request body and put it on the Kinesis stream (202 Accepted)."""
    feedback_event = json.loads(event.get("body") or "{}")

    missing_fields = [field for field in ("user_id", "event_type") if not feedback_event.get(field)]
    if missing_fields:
        return response(400, {"detail": f"Missing fields: {missing_fields}"})
    if feedback_event["event_type"] not in ALLOWED_EVENT_TYPES:
        return response(400, {"detail": f"Unknown event_type: {feedback_event['event_type']}"})

    # Fill server-side id/timestamp only if the client didn't send them.
    feedback_event.setdefault("event_id", f"evt-{uuid.uuid4().hex[:12]}")
    feedback_event.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Partitioning by user_id keeps each user's events in order within one shard.
    kinesis.put_record(
        StreamName=STREAM_NAME,
        Data=json.dumps(feedback_event).encode("utf-8"),
        PartitionKey=feedback_event["user_id"],
    )
    return response(202, {"accepted": True, "event_id": feedback_event["event_id"]})
