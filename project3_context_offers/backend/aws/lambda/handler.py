"""API Gateway -> Lambda -> SageMaker endpoint (guide section 31).

Route: GET /v1/offers/recommendations/{user_id}?device_type=&region=&hour=&day_of_week=&limit=
The frontend supplies the context; this handler turns it into the inference
request body. The browser never holds AWS credentials.
"""
import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")


def response(status_code, body):
    """API Gateway proxy response with a CORS header; `body` may already be a JSON string."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": ALLOWED_ORIGIN},
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def lambda_handler(event, context):
    """Turn the path/query parameters into an inference request and return the endpoint's JSON."""
    path_parameters = event.get("pathParameters") or {}
    query = event.get("queryStringParameters") or {}
    user_id = path_parameters.get("user_id")
    if not user_id:
        return response(400, {"detail": "user_id is required"})

    try:
        request_body = {
            "user_id": user_id,
            "context": {
                "device_type": query.get("device_type", "web"),
                "region": query.get("region", "Lagos"),
                "hour": int(query.get("hour", 12)),
                "day_of_week": int(query.get("day_of_week", 0)),
            },
            "limit": int(query.get("limit", 10)),
        }
    except ValueError:
        return response(400, {"detail": "hour, day_of_week and limit must be integers"})

    # The endpoint's body is JSON already (see inference.output_fn), so it is passed through as-is.
    sagemaker_response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(request_body),
    )
    return response(200, sagemaker_response["Body"].read().decode("utf-8"))
