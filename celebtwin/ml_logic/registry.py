import glob
import os
import pickle
from pathlib import Path

import keras
from celebtwin.params import LOCAL_REGISTRY_PATH, MODEL_TARGET, BUCKET_NAME
from colorama import Fore, Style
from google.cloud import storage


def save_metadata(name: str, metadata: dict) -> None:
    """Persist parameters, metrics and history locally.

    Save metadata in {LOCAL_REGISTRY_PATH}/metadata.

    If MODEL_TARGET='gcs', also upload to GCS in metadata folder.
    """
    registry = Path(LOCAL_REGISTRY_PATH)
    metadata_dir = registry / 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = metadata_dir / (name + ".pickle")
    with open(metadata_path, "wb") as file:
        pickle.dump(metadata, file)
    print("✅ Results saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"metadata/{name}.pickle")
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


def load_model() -> keras.Model:
    """Return a saved model.

    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found
    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + "Load latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + "Load latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + "Load latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(
                LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except Exception:
            print(f"❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
