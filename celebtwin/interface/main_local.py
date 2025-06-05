import click
import numpy as np
import pandas as pd
from celebtwin.ml_logic.data import ColorMode, ResizeMode, SimpleDataset
from celebtwin.ml_logic.experiment import Experiment
from celebtwin.ml_logic.model import SimpleLeNetModel
from celebtwin.ml_logic.registry import load_model
from colorama import Fore, Style


@click.group()
def cli():
    pass


@cli.command()
def train():
    """Train on a local dataset.

    Save validation metrics and the trained model.
    """
    print(Fore.MAGENTA + " ⭐️ Training" + Style.RESET_ALL)

    image_size = 64
    num_classes = 2
    color_mode = ColorMode.GRAYSCALE
    batch_size = 256
    validation_split = 0.2
    learning_rate = 0.001
    patience = 5

    dataset = SimpleDataset(
        image_size=image_size,
        num_classes=num_classes,
        undersample=False,
        color_mode=color_mode,
        resize=ResizeMode.PAD,
        batch_size=batch_size,
        validation_split=validation_split)
    model = SimpleLeNetModel(
        input_shape=(image_size, image_size, color_mode.num_channels()),
        class_nb=num_classes)
    experiment = Experiment(
        dataset=dataset,
        model=model,
        learning_rate=learning_rate,
        patience=patience)
    experiment.run()
    experiment.save_results()
    print("✅ train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("✅ pred() done")

    return y_pred


if __name__ == '__main__':
    try:
        cli()
    except Exception:
        import sys
        import traceback
        import ipdb  # type: ignore
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
