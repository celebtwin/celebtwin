import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from celebtwin.ml_logic.encoders import transform_time_features, transform_lonlat_features, compute_geohash


def preprocess_features(X) -> np.ndarray:
    X_processed = X

    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed
