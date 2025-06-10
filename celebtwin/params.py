import os

GCP_PROJECT = "celebtwin"
BQ_DATASET = "celebtwin"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "celebtwin")
MODEL_TARGET = "gcs"  # "local" ou "gcs"

LOCAL_DATA_PATH = os.path.join(
    os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH = "training_outputs"
