"""API Gateway -> Lambda -> Kinesis (guide sections 35 and 40).

Route: POST /v1/financial-products/events
The request ID and strategy answer "what did the system recommend before the
user acted?". No financial-profile data is written to the stream.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import boto3

kinesis = boto3.client("kinesis")
STREAM_NAME = os.environ["FEEDBACK_STREAM_NAME"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
ALLOWED_EVENT_TYPES = {"recommendation_impression", "recommendation_click", "view", "learn", "apply"}
ALLOWED_FIELDS = {"event_id", "request_id", "user_id", "product_id", "event_type", "recommendation_strategy", "timestamp"}


def response(status_code, body):
    """API Gateway proxy response with JSON body and CORS header."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": ALLOWED_ORIGIN},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """Validate the event, keep only allowed fields and put it on the Kinesis stream (202)."""
    submitted = json.loads(event.get("body") or "{}")
    missing_fields = [field for field in ("user_id", "product_id", "event_type") if not submitted.get(field)]
    if missing_fields:
        return response(400, {"detail": f"Missing fields: {missing_fields}"})
    if submitted["event_type"] not in ALLOWED_EVENT_TYPES:
        return response(400, {"detail": f"Unknown event_type: {submitted['event_type']}"})

    # Data minimisation: only the schema's fields are forwarded.
    feedback_event = {field: value for field, value in submitted.items() if field in ALLOWED_FIELDS}
    feedback_event.setdefault("event_id", f"evt-{uuid.uuid4().hex[:12]}")
    feedback_event.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # Partition by user so one user's events keep their order within a shard.
    kinesis.put_record(
        StreamName=STREAM_NAME,
        Data=json.dumps(feedback_event).encode("utf-8"),
        PartitionKey=feedback_event["user_id"],
    )
    return response(202, {"accepted": True, "event_id": feedback_event["event_id"]})
