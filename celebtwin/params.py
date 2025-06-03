import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = "1k" # ["1k", "200k", "all"]
CHUNK_SIZE = 200
GCP_PROJECT = "celebtwin"
BQ_DATASET = "celebtwin"
BQ_REGION = "EU"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "celebtwin")
MODEL_TARGET = "local" #ou "gcs"
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH = "training_outputs"
