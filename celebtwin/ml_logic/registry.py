import glob
import json
import os
from pathlib import Path

import keras  # type: ignore
from celebtwin.params import BUCKET_NAME, LOCAL_REGISTRY_PATH, MODEL_TARGET
from google.cloud import storage  # type: ignore


def save_metadata(name: str, metadata: dict) -> None:
    """Persist parameters, metrics and history locally.

    Save metadata in {LOCAL_REGISTRY_PATH}/metadata.

    If MODEL_TARGET='gcs', also upload to GCS in metadata folder.
    """
    registry = Path(LOCAL_REGISTRY_PATH)
    metadata_dir = registry / 'metadata'
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
    models_dir = Path(LOCAL_REGISTRY_PATH) / "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / f"{identifier}.keras"
    model.save(str(model_path))
    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/staging/{model_path.name}")
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
        return f"❌ No model found in GCS bucket {BUCKET_NAME}."


def load_model() -> keras.Model:
    """Return a saved model.

    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Raises NoModelFoundError if no model is found in the registry.
    """
    if MODEL_TARGET == "local":
        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")
        if not local_model_paths:
            raise NoLocalModelFoundError()
        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        latest_model = keras.models.load_model(most_recent_model_path_on_disk)
        print("✅ Model loaded from local disk")
        return latest_model  # type: ignore

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        if not blobs:
            raise NoGCSModelFoundError()
        latest_blob = max(blobs, key=lambda x: x.updated)

        # Download to temporary file, then rename file and load the model.
        latest_model_path_to_save = os.path.join(
            LOCAL_REGISTRY_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        latest_model = keras.models.load_model(latest_model_path_to_save)
        print("✅ Latest model downloaded from cloud storage")
        return latest_model  # type: ignore

    raise ValueError(
        f"MODEL_TARGET must be 'local' or 'gcs', got {MODEL_TARGET}")
