import json, os, boto3
runtime=boto3.client("sagemaker-runtime")
ENDPOINT=os.environ["SAGEMAKER_ENDPOINT"]

def handler(event, context):
    body=json.loads(event.get("body") or "{}")
    response=runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(body).encode()
    )
    return {"statusCode":200,
            "headers":{"content-type":"application/json"},
            "body":response["Body"].read().decode()}
