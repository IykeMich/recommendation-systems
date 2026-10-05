#!/usr/bin/env bash
# Build model.tar.gz for a SageMaker scikit-learn container (guide section 19).
# Usage: aws/sagemaker/package.sh s3://YOUR-BUCKET/models/fraud/v1/
# Stop on any error, unset variable or failed pipe stage.
set -euo pipefail
# Run from backend/ (the folder holding src/ and artifacts/) regardless of where the script is called from.
cd "$(dirname "$0")/../.."
# Layout SageMaker expects: model files at the archive root, inference code under code/.
BUILD=$(mktemp -d)
mkdir -p "$BUILD/code"
cp artifacts/fraud_pipeline.joblib artifacts/feature_manifest.json "$BUILD/"
cp aws/sagemaker/inference.py aws/sagemaker/requirements.txt "$BUILD/code/"
# src/ is needed at inference time: the pickled artifact references src.* classes.
cp -R src "$BUILD/code/src"
find "$BUILD/code/src" -name __pycache__ -prune -exec rm -rf {} +
tar -czf model.tar.gz -C "$BUILD" .
echo "built model.tar.gz ($(du -h model.tar.gz | cut -f1))"
# Optional upload: the S3 prefix argument must end with "/" (the file name is appended).
if [ "${1:-}" != "" ]; then
  aws s3 cp model.tar.gz "$1model.tar.gz"
fi
