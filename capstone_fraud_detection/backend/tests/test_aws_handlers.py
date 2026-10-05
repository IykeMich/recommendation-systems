"""AWS handlers, tested locally: the real SageMaker handler on the trained artifact,
and the Lambdas with in-memory fakes for boto3 (DynamoDB, SageMaker runtime)."""
import base64
import importlib
import io
import json
import sys
import types

import pytest

from src.config import ARTIFACT_DIR

TRANSACTION = {"transaction_id": "TXN-AWS-1", "customer_id": "F00001", "amount": 1500, "currency": "NGN",
               "hour": 3, "velocity_1h": 9, "country": "NG", "device": "android", "merchant_category": "travel"}


def load_module(name, path):
    """Import a module from a file path (the aws/ folders are not Python packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sagemaker_handler_round_trip():
    inference = load_module("inference", "aws/sagemaker/inference.py")
    scorer = inference.model_fn(str(ARTIFACT_DIR))
    body, content_type = inference.output_fn(inference.predict_fn(inference.input_fn(json.dumps(TRANSACTION)), scorer))
    prediction = json.loads(body)
    assert content_type == "application/json" and 0 <= prediction["risk_score"] <= 1
    with pytest.raises(ValueError):
        inference.input_fn(json.dumps({**TRANSACTION, "amount": -1}))


class ConditionalCheckFailed(Exception):
    """Stands in for botocore's ClientError carrying a ConditionalCheckFailedException code."""
    def __init__(self):
        self.response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class FakeTable:
    """In-memory DynamoDB table that honours the attribute_not_exists conditional put."""
    def __init__(self):
        self.items = {}

    def get_item(self, Key):
        item = self.items.get(Key["transaction_id"])
        return {"Item": item} if item else {}

    def put_item(self, Item, ConditionExpression):
        if Item["transaction_id"] in self.items:
            raise ConditionalCheckFailed()
        self.items[Item["transaction_id"]] = Item


@pytest.fixture
def lambdas(monkeypatch):
    """Import the Lambda modules against fake boto3/botocore and env vars; yields
    (score_api, stream_consumer, list of endpoint invocations).
    """
    table = FakeTable()
    invocations = []

    def invoke_endpoint(EndpointName, ContentType, Body):
        invocations.append(json.loads(Body))
        return {"Body": io.BytesIO(json.dumps({"risk_score": 0.71, "model_version": "fraud-test"}).encode())}

    fake_boto3 = types.SimpleNamespace(
        client=lambda name: types.SimpleNamespace(invoke_endpoint=invoke_endpoint),
        resource=lambda name: types.SimpleNamespace(Table=lambda table_name: table),
    )
    fake_exceptions = types.SimpleNamespace(ClientError=ConditionalCheckFailed)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", types.SimpleNamespace(exceptions=fake_exceptions))
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_exceptions)
    for name, value in {"DECISIONS_TABLE": "t", "SAGEMAKER_ENDPOINT": "e"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.syspath_prepend("aws/lambda")
    for module_name in ("fraud_common", "score_api", "stream_consumer"):
        sys.modules.pop(module_name, None)
    yield importlib.import_module("score_api"), importlib.import_module("stream_consumer"), invocations
    for module_name in ("fraud_common", "score_api", "stream_consumer"):
        sys.modules.pop(module_name, None)


def test_api_lambda_applies_policy_and_is_idempotent(lambdas):
    score_api, _, invocations = lambdas
    first = json.loads(score_api.lambda_handler({"body": json.dumps(TRANSACTION)}, None)["body"])
    second = json.loads(score_api.lambda_handler({"body": json.dumps(TRANSACTION)}, None)["body"])
    assert first["decision"] == "REVIEW" and first["policy_version"] == "policy-v1"
    assert second["idempotent_replay"] is True and len(invocations) == 1


def test_api_lambda_rejects_bad_input(lambdas):
    score_api, _, invocations = lambdas
    assert score_api.lambda_handler({"body": json.dumps({**TRANSACTION, "amount": 0})}, None)["statusCode"] == 422
    assert score_api.lambda_handler({"body": json.dumps({**TRANSACTION, "risk_score": 0})}, None)["statusCode"] == 422
    assert score_api.lambda_handler({"body": "not json"}, None)["statusCode"] == 400
    assert invocations == []


def test_stream_lambda_skips_malformed_and_dedupes(lambdas):
    _, stream_consumer, invocations = lambdas
    encode = lambda payload: base64.b64encode(json.dumps(payload).encode()).decode()
    event = {"Records": [
        {"kinesis": {"sequenceNumber": "1", "data": encode(TRANSACTION)}},
        {"kinesis": {"sequenceNumber": "2", "data": encode(TRANSACTION)}},             # redelivery
        {"kinesis": {"sequenceNumber": "3", "data": encode({"transaction_id": "bad"})}},  # malformed
    ]}
    assert stream_consumer.lambda_handler(event, None) == {"batchItemFailures": []}
    assert len(invocations) == 1
