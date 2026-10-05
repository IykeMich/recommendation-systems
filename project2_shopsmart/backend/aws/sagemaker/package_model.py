"""Build the S3 upload bundle locally (guide §17): data, training code, model and metrics.

Run from backend/: python aws/sagemaker/package_model.py
Output goes to aws/build/, mirroring the S3 layout:

    aws/build/data/{users,products,interactions}.csv
    aws/build/code/sourcedir.tar.gz      <- training code for a SageMaker AI training job
    aws/build/models/model.tar.gz        <- a locally trained model (alternative to the training job)
    aws/build/evaluation/metrics.json

sourcedir.tar.gz holds train.py and inference.py at the top level next to src/, which is the
layout a SageMaker framework container expects. Point a training job at it with the hyperparameters
sagemaker_program=train.py and sagemaker_submit_directory=s3://.../code/sourcedir.tar.gz. It has no
requirements.txt on purpose: the container's own pandas/NumPy/scikit-learn are used, so the model
it trains loads in the same container image used for serving.

It runs aws/sagemaker/train.py exactly as a SageMaker training job would (same SM_* variables,
training channel = the three CSVs), then packs the model directory in SageMaker's layout:

    model.tar.gz
    ├── model.joblib, popular_items.json, model_metadata.json   <- model_fn reads these
    └── code/
        ├── inference.py                                         <- entry point
        ├── requirements.txt                                     <- pinned to the versions that built the model
        └── src/                                                 <- the pickled model imports src.recommender

The pickle only loads under compatible pandas/NumPy/SciPy/scikit-learn versions, hence the pinned
requirements.txt. The most robust alternative is to run train.py as a real SageMaker training job,
so the same container image trains and serves the model.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from importlib.metadata import version
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
BUILD = BACKEND / "aws" / "build"
TRAINING_FILES = ["users.csv", "products.csv", "interactions.csv"]
PINNED_PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]


def build_source_dir(archive_path: Path) -> None:
    """Pack train.py + inference.py + src/ (no caches) for a SageMaker AI training job."""
    with tarfile.open(archive_path, "w:gz") as archive:
        for script in ["train.py", "inference.py"]:
            archive.add(BACKEND / "aws" / "sagemaker" / script, arcname=script)
        for path in sorted((BACKEND / "src").rglob("*.py")):
            archive.add(path, arcname=str(path.relative_to(BACKEND)))


def main():
    shutil.rmtree(BUILD, ignore_errors=True)
    data_dir, models_dir, evaluation_dir = BUILD / "data", BUILD / "models", BUILD / "evaluation"
    code_dir = BUILD / "code"
    for directory in (data_dir, models_dir, evaluation_dir, code_dir):
        directory.mkdir(parents=True)
    build_source_dir(code_dir / "sourcedir.tar.gz")

    # The training channel holds only the original CSVs (no new_interactions.csv, no archive/).
    for name in TRAINING_FILES:
        shutil.copy2(BACKEND / "data" / name, data_dir / name)

    with tempfile.TemporaryDirectory() as workdir:
        model_dir = Path(workdir) / "model"
        output_dir = Path(workdir) / "output"
        environment = {
            **os.environ,
            "SM_CHANNEL_TRAINING": str(data_dir),
            "SM_MODEL_DIR": str(model_dir),
            "SM_OUTPUT_DATA_DIR": str(output_dir),
        }
        subprocess.run([sys.executable, "aws/sagemaker/train.py"], cwd=BACKEND, env=environment, check=True)
        shutil.copy2(output_dir / "metrics.json", evaluation_dir / "metrics.json")

        code_dir = model_dir / "code"
        shutil.copytree(BACKEND / "src", code_dir / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(BACKEND / "aws" / "sagemaker" / "inference.py", code_dir / "inference.py")
        (code_dir / "requirements.txt").write_text(
            "".join(f"{package}=={version(package)}\n" for package in PINNED_PACKAGES)
        )

        with tarfile.open(models_dir / "model.tar.gz", "w:gz") as archive:
            for path in sorted(model_dir.rglob("*")):
                archive.add(path, arcname=str(path.relative_to(model_dir)), recursive=False)

    for path in sorted(BUILD.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(BACKEND)}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
