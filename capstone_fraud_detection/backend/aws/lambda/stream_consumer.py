"""Kinesis Data Streams → Lambda (event source mapping) (guide section 18).

Records can be delivered more than once; fraud_common makes scoring idempotent
by transaction_id. Malformed records are logged and skipped (not retried
forever); transient failures are reported as batch item failures so only those
records are retried.
"""
import base64
import json

from fraud_common import InvalidTransaction, score_and_decide


def lambda_handler(event, context):
    """Score each Kinesis record in the batch; return the sequence numbers that should be retried."""
    failures = []
    for record in event.get("Records", []):
        sequence_number = record["kinesis"]["sequenceNumber"]
        try:
            # Kinesis delivers the record payload base64-encoded.
            transaction = json.loads(base64.b64decode(record["kinesis"]["data"]))
            score_and_decide(transaction)
        except (InvalidTransaction, json.JSONDecodeError) as error:
            print(json.dumps({"event": "rejected", "sequence_number": sequence_number, "reason": str(error)}))
        except Exception as error:  # transient (throttling, endpoint timeout): retry this record
            print(json.dumps({"event": "retry", "sequence_number": sequence_number, "reason": repr(error)}))
            failures.append({"itemIdentifier": sequence_number})
    # Partial batch response (needs ReportBatchItemFailures on the event source mapping): Lambda
    # retries from the earliest failed sequence number instead of the whole batch. An empty list
    # means the whole batch succeeded.
    return {"batchItemFailures": failures}
