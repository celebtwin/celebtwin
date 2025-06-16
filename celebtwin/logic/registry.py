import json
import os
import subprocess
from glob import glob
from pathlib import Path

import keras  # type: ignore
from google.cloud import storage  # type: ignore
from google.cloud.exceptions import NotFound

from celebtwin.logic import experiment
from celebtwin.params import BUCKET_NAME, LOCAL_REGISTRY_PATH, MODEL_TARGET

models_dir = Path(LOCAL_REGISTRY_PATH) / "models"
metadata_dir = Path(LOCAL_REGISTRY_PATH) / "metadata"


def save_metadata(name: str, metadata: dict) -> None:
    """Persist parameters, metrics and history locally.

    Save metadata in {LOCAL_REGISTRY_PATH}/metadata.

    If MODEL_TARGET='gcs', also upload to GCS in metadata folder.
    """
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = metadata_dir / (name + ".json")
    with open(metadata_path, "wt", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
        print("✅ Metadata saved locally")
    print("✅ Results saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"metadata/{name}.json")
        blob.upload_from_filename(metadata_path)
        print("✅ Results saved to GCS")


def save_model(model: keras.Model, identifier: str):
    """Save trained model locally, and optionally on GCS.

    Save model in {LOCAL_REGISTRY_PATH}/models.

    If MODEL_TARGET='gcs', also upload to GCS in models/staging folder.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / f"{identifier}.keras"
    model.save(str(model_path))
    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/staging/{model_path.name}")
        print("☁️ Uploading model to GCS: " + model_path.name)
        blob.upload_from_filename(model_path)
        print("✅ Model saved to GCS")


class NoModelFoundError(Exception):
    """Exception raised when no model is found in the registry."""
    pass


class NoLocalModelFoundError(NoModelFoundError):
    def __str__(self):
        return "❌ No model found in local registry."


class NoGCSModelFoundError(NoModelFoundError):
    def __str__(self):
        return f"❌ No production model found in GCS bucket {BUCKET_NAME}."


def load_latest_experiment() -> 'experiment.Experiment':
    """Return the latest experiment from disk or the production from GCS.

    If MODEL_TARGET is 'gcs', find the latest experiment in production, download it if needed, and return it. The experiment is cached locally.

    If MODEL_TARGET is 'local', find the latest experiment on disk and return it.
    """
    from celebtwin.logic.experiment import load_experiment

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs(
            prefix="models/production/"))
        if not blobs:
            raise NoGCSModelFoundError()
        latest_blob = max(blobs, key=lambda x: x.updated)
        model_name = latest_blob.name.split("/")[-1]
        assert model_name.endswith(".keras"), \
            f"Expected model name to end with '.keras', got {model_name}"
        metadata_name = model_name.replace(".keras", ".json")
        model_path = models_dir / model_name
        metadata_path = metadata_dir / metadata_name
        if not (model_path).exists():
            os.makedirs(models_dir, exist_ok=True)
            print("☁️ Downloading model from GCS: " + model_name)
            latest_blob.download_to_filename(model_path)
            os.makedirs(metadata_dir, exist_ok=True)
            bucket.blob(f"metadata/{metadata_name}")\
                .download_to_filename(metadata_path)
            print("✅ Latest model downloaded from cloud storage")
        return load_experiment(metadata_path, model_path)

    if MODEL_TARGET == "local":
        # Find latest local model based on the timestamp in the filename.
        model_names = glob("*.keras", root_dir=models_dir)
        if not model_names:
            raise NoLocalModelFoundError()
        model_path = models_dir / list(sorted(model_names))[-1]
        metadata_path = metadata_dir / f"{model_path.stem}.json"
        return load_experiment(metadata_path, model_path)

    assert False, \
        f"Unknown MODEL_TARGET: {MODEL_TARGET}. Expected 'gcs' or 'local'."


def try_download_dataset(path: Path) -> bool:
    """Try to download and unzip a dataset from GCS.

    Args:
        path: Path where the dataset should be extracted

    Returns:
        True if the dataset was downloaded and extracted, False if it doesn't exist
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    zip_path = path.with_suffix('.zip')
    blob = bucket.blob('dataset/' + zip_path.name)
    try:
        blob.reload()
    except NotFound:
        return False
    print("☁️ Downloading dataset from GCS: " + zip_path.name)
    blob.download_to_filename(zip_path)
    print("✅ Dataset downloaded from GCS")
    subprocess.run(['unzip', '-q', zip_path.name],
                   cwd=path.parent, check=True)
    zip_path.unlink()
    return True


def upload_dataset(path: Path) -> None:
    """Create a zip file for a dataset and upload it to GCS.

    Args:
        path: Path to the dataset directory to upload
    """
    if MODEL_TARGET == 'local':
        return
    zip_path = path.with_suffix('.zip')
    subprocess.run(['zip', '-q', '-r', zip_path.name, path.name],
                   cwd=path.parent, check=True)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob('dataset/' + zip_path.name)
    print("☁️ Uploading dataset to GCS: " + zip_path.name)
    blob.upload_from_filename(zip_path)
    print("✅ Dataset uploaded to GCS")
    zip_path.unlink()
