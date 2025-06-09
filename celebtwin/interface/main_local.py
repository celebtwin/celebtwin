import sys

import click
from colorama import Fore, Style


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '--dataset', type=click.Choice(['aligned', 'simple']), default='simple',
    help='Dataset preprocessing, defaults to simple.')
@click.option(
    '--model', type=click.Choice(['simple', 'weekend']), default='simple',
    help='Model to train, defaults to simple.')
def train(dataset, model) -> None:
    """Train on a local dataset.

    Save validation metrics and the trained model.
    """
    print(Fore.BLUE + "Starting up" + Style.RESET_ALL)
    from celebtwin.ml_logic.data import (
        AlignedDataset, ColorMode, ResizeMode, SimpleDataset)
    from celebtwin.ml_logic.experiment import Experiment
    from celebtwin.ml_logic.model import SimpleLeNetModel, WeekendModel
    print(Fore.MAGENTA + "⭐️ Training" + Style.RESET_ALL)

    image_size = 64
    num_classes = 5
    color_mode = ColorMode.RGB
    batch_size = 256
    validation_split = 0.2
    learning_rate = 0.00001
    patience = 10

    dataset_class = {
        'simple': SimpleDataset, 'aligned': AlignedDataset}[dataset]
    dataset_instance = dataset_class(
        image_size=image_size,
        num_classes=num_classes,
        undersample=False,
        color_mode=color_mode,
        resize=ResizeMode.PAD,
        batch_size=batch_size,
        validation_split=validation_split)

    # Annoying code needed to avoid type errors when building the model.
    model_instance: SimpleLeNetModel | WeekendModel
    if model == 'simple':
        model_instance = SimpleLeNetModel()
    elif model == 'weekend':
        model_instance = WeekendModel()
    else:
        raise AssertionError(f"Invalid model: {model}")

    model_instance.build(
        input_shape=(image_size, image_size, color_mode.num_channels()),
        class_nb=num_classes)
    experiment = Experiment(
        dataset=dataset_instance,
        model=model_instance,
        learning_rate=learning_rate,
        patience=patience)
    experiment.run()
    experiment.save_results()
    print("✅ train() done")


@cli.command()
def batch_align() -> None:
    """Perform face alignment on all images."""
    print(Fore.BLUE + "Starting up" + Style.RESET_ALL)
    from celebtwin.ml_logic.data import AlignedDatasetFull
    print(Fore.MAGENTA + "⭐️ Batch aligning" + Style.RESET_ALL)
    AlignedDatasetFull().preprocess_all()
    print("✅ batch_align() done")


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def pred(image_path) -> None:
    """Predict the class of a single image."""
    print(Fore.MAGENTA + "⭐️ Predicting" + Style.RESET_ALL)
    from celebtwin.ml_logic.preproc_face import NoFaceDetectedError
    from celebtwin.ml_logic.registry import (
        NoModelFoundError, load_latest_experiment)
    try:
        experiment = load_latest_experiment()
    except NoModelFoundError as error:
        print(Fore.RED + str(error) + Style.RESET_ALL)
        print(Fore.YELLOW + "Please train a model first." + Style.RESET_ALL)
        sys.exit(1)
    try:
        pred_probas, class_name = experiment.predict(image_path)
    except NoFaceDetectedError as error:
        print(Fore.RED + str(error) + Style.RESET_ALL)
        sys.exit(1)
    print(Fore.GREEN + f"Predicted class: {class_name}" + Style.RESET_ALL)
    print(Fore.BLUE + f"Predicted probabilities: {pred_probas}"
          + Style.RESET_ALL)


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
