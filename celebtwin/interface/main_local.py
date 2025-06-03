import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from celebtwin.params import *
from celebtwin.ml_logic.data import load_dataset
from celebtwin.ml_logic.preprocessor import preprocess_features
from celebtwin.ml_logic.registry import save_model, save_results, load_model
from celebtwin.ml_logic.model import compile_model, initialize_model, train_model

def preprocess_and_train() -> None:
    """
    - Query the raw dataset from ...
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Train a Keras model on it
    - Save the model
    - Compute & save a validation performance metric
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """

    # Retrieve `query` data from BigQuery or from `data_query_cache_path` if the file already exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")

        # $CODE_BEGIN
        data = pd.read_csv(data_query_cache_path)
        # $CODE_END

    else:
        print("Loading data from Querying Big Query server...")

        # $CODE_BEGIN
        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        data = result.to_dataframe()
        # $CODE_END

        # Save it locally to accelerate the next queries!
        data.to_csv(data_query_cache_path, header=True, index=False)

    dataset_loader = make_dataset_loader(
        image_size=64, num_classes=2, undersample=True, color_mode='grayscale',
        resize='pad')

    # We do not need to clean data. But we may want to apply further image
    # preprocessing. That should be done either in the model pipeline or, if we
    # want to save on duplicate processing, in the dataset generation.

    #data = clean_data(data)

    split_ratio = 0.02 # About one month of validation data

    # TODO: Remove this code. The train-test split is handled by
    # keras.preprocessing.image_dataset_from_directory.

    train_length = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:train_length, :].sample(frac=1)
    data_val = data.iloc[train_length:, :].sample(frac=1)

    X_train = data_train.drop("fare_amount", axis=1)
    y_train = data_train[["fare_amount"]]

    X_val = data_val.drop("fare_amount", axis=1)
    y_val = data_val[["fare_amount"]]

    # TODO: Remove preprocess_features. Preprocessing should by done either in
    # the model or during dataset generation.
    X_train_processed = preprocess_features(X_train)
    X_val_processed = preprocess_features(X_val)

    # Train a model on the training set, using `model.py`
    model = None
    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    # $CODE_BEGIN
    model = initialize_model(input_shape=X_train_processed.shape[1:])
    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model, X_train_processed, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val)
    )
    # $CODE_END

    # Compute the validation metric (min val_mae) of the holdout set
    val_mae = np.min(history.history['val_mae'])

    # Save trained model
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")

# $ERASE_BEGIN
def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    1. Query and preprocess the raw dataset iteratively (in chunks)
    2. Store the newly processed (and raw) data on your local hard drive for later use

    - If raw data already exists on your local disk, use `pd.read_csv(..., chunksize=CHUNK_SIZE)`
    - If raw data does NOT yet exist, use `bigquery.Client().query().result().to_dataframe_iterable()`
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    # Retrieve `query` data as a DataFrame iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    if data_query_cache_exists:
        print("Get a DataFrame iterable from local CSV...")
        chunks = None

        # $CODE_BEGIN
        chunks = pd.read_csv(
            data_query_cache_path,
            chunksize=CHUNK_SIZE,
            parse_dates=["pickup_datetime"]
        )
        # $CODE_END
    else:
        print("Get a DataFrame iterable from querying the BigQuery server...")
        chunks = None

        # 🎯 HINT: `bigquery.Client(...).query(...).result(page_size=...).to_dataframe_iterable()`
        # $CODE_BEGIN
        client = bigquery.Client(project=GCP_PROJECT)

        query_job = client.query(query)
        result = query_job.result(page_size=CHUNK_SIZE)

        chunks = result.to_dataframe_iterable()
        # $CODE_END

    for chunk_id, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_id}...")

        # Clean chunk
        # $CODE_BEGIN
        chunk_clean = clean_data(chunk)
        # $CODE_END

        # Create chunk_processed
        # 🎯 HINT: create (`X_chunk`, `y_chunk`), process only `X_processed_chunk`, then concatenate (X_processed_chunk, y_chunk)
        # $CODE_BEGIN
        X_chunk = chunk_clean.drop("fare_amount", axis=1)
        y_chunk = chunk_clean[["fare_amount"]]
        X_processed_chunk = preprocess_features(X_chunk)

        chunk_processed = pd.DataFrame(np.concatenate((X_processed_chunk, y_chunk), axis=1))
        # $CODE_END

        # Save and append the processed chunk to a local CSV at "data_processed_path"
        # 🎯 HINT: df.to_csv(mode=...)
        # 🎯 HINT: we want a CSV with neither index nor headers (they'd be meaningless)
        # $CODE_BEGIN
        chunk_processed.to_csv(
            data_processed_path,
            mode="w" if chunk_id==0 else "a",
            header=False,
            index=False,
        )
        # $CODE_END

        # Save and append the raw chunk `if not data_query_cache_exists`
        # $CODE_BEGIN
        # 🎯 HINT: we want a CSV with headers this time
        # 🎯 HINT: only the first chunk should store headers
        if not data_query_cache_exists:
            chunk.to_csv(
                data_query_cache_path,
                mode="w" if chunk_id==0 else "a",
                header=True if chunk_id==0 else False,
                index=False
            )
        # $CODE_END

    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")


def train():
    """
    Train on a local dataset.

    Save validation metrics and the trained model.
    """
    print(Fore.MAGENTA + " ⭐️ Training" + Style.RESET_ALL)

    image_size = 64
    num_classes = 2
    color_mode = 'grayscale'
    batch_size = 256
    validation_split = 0.2
    learning_rate = 0.001
    patience = 5

    train_dataset, val_dataset = load_dataset(
        image_size,
        num_classes,
        undersample=False,
        color_mode=color_mode,
        resize='pad',
        batch_size=batch_size,
        validation_split=validation_split
    )
    assert len(train_dataset.class_names) == num_classes

    n_channels = {'grayscale': 1, 'color': 3}[color_mode]
    model = initialize_model(
        input_shape=(image_size, image_size),
        class_nb=len(train_dataset.class_names),
        colors=(color_mode == 'color'))
    model = compile_model(model, learning_rate)
    model, history = train_model(
        model, train_dataset, val_dataset, patience)

    val_accuracy = np.max(history['val_accuracy'])
    save_model(model=model)

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience)
    save_results(params=params, metrics=dict(accuracy=val_accuracy))

    print("✅ train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")

    return y_pred


if __name__ == '__main__':
    try:
        #preprocess_and_train()
        #preprocess()
        #train()
        #pred()
        pass
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
