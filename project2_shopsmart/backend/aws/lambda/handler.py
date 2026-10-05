"""API Gateway -> Lambda -> SageMaker endpoint (sections 20-21).

Route: GET /v1/recommendations/user/{user_id}?limit=10
API Gateway passes path/query parameters; this handler turns them into the
SageMaker inference request body. The browser never holds AWS credentials.
"""
import json
import os

import boto3

# Created once per Lambda container (outside the handler) so warm invocations reuse the client.
runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT"]
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")


def response(status_code, body):
    """Build an API Gateway proxy response; a str body is passed through as already-encoded JSON."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        },
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def lambda_handler(event, context):
    """Read user_id/limit from the path, query string or JSON body and forward them to SageMaker."""
    path_parameters = event.get("pathParameters") or {}
    query_parameters = event.get("queryStringParameters") or {}
    body = json.loads(event.get("body") or "{}")

    user_id = path_parameters.get("user_id") or body.get("user_id")
    if not user_id:
        return response(400, {"detail": "user_id is required"})

    try:
        limit = int(query_parameters.get("limit") or body.get("limit") or 10)
    except ValueError:
        return response(400, {"detail": "limit must be an integer"})

    # The endpoint already returns JSON, so its body is relayed to the caller unchanged.
    sagemaker_response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"user_id": user_id, "limit": limit}),
    )
    return response(200, sagemaker_response["Body"].read().decode("utf-8"))
