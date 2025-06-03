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
