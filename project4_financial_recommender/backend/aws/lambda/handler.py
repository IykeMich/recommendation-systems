"""API Gateway -> Lambda -> SageMaker endpoint (guide section 39).

Route: GET /v1/financial-products/recommendations/{user_id}?limit=5
Least privilege: the role needs only sagemaker:InvokeEndpoint on this endpoint.
"""
import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")


def response(status_code, body):
    """API Gateway proxy response with CORS header; a str body is passed through unchanged."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": ALLOWED_ORIGIN},
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def lambda_handler(event, context):
    """Validate user_id/limit, then forward the request to the SageMaker endpoint."""
    user_id = (event.get("pathParameters") or {}).get("user_id")
    if not user_id:
        return response(400, {"detail": "user_id is required"})
    try:
        limit = int((event.get("queryStringParameters") or {}).get("limit", 5))
    except ValueError:
        return response(400, {"detail": "limit must be an integer"})

    sagemaker_response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"user_id": user_id, "limit": limit}),
    )
    return response(200, sagemaker_response["Body"].read().decode("utf-8"))
