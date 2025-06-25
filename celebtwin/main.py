import os
import sys
from pathlib import Path

import click
from colorama import Fore, Style

from . import logic
from .logic.annenums import Detector, Model, ANNBackend


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    '-d', '--dataset', type=click.Choice(['aligned', 'simple']), default='simple',
    help='Dataset preprocessing, defaults to simple.')
@click.option(
    '-m', '--model', type=click.Choice(['simple', 'weekend']), default='simple',
    help='Model to train, defaults to simple.')
@click.option(
    '-c', '--classes', type=int,
    help='Number of classes in the model, or "all", default to 2.')
@click.option(
    '-l', '--learning-rate', type=float, default=0.001,
    help='Learning rate, defaults to 0.001.')
@click.option(
    '-p', '--patience', type=int, default=10,
    help='Stop after this many epochs without improvement, defaults to 10.')
def train(dataset: str, model: str, classes: int, learning_rate: float,
          patience: int) -> None:
    """Train on a local dataset.

    Save validation metrics and the trained model.
    """
    print(Fore.BLUE + "Starting up" + Style.RESET_ALL)
    from celebtwin.logic.data import (
        AlignedDataset, ColorMode, ResizeMode, SimpleDataset)
    from celebtwin.logic.experiment import Experiment
    from celebtwin.logic.model import SimpleLeNetModel, WeekendModel
    print(Fore.MAGENTA + "⭐️ Training" + Style.RESET_ALL)

    image_size = 64
    color_mode = ColorMode.RGB
    batch_size = 256
    validation_split = 0.2

    dataset_class = {
        'simple': SimpleDataset, 'aligned': AlignedDataset}[dataset]
    dataset_instance = dataset_class(
        image_size=image_size,
        num_classes=classes,
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
        class_nb=classes)
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
    from celebtwin.logic.data import AlignedDatasetFull
    print(Fore.MAGENTA + "⭐️ Batch aligning" + Style.RESET_ALL)
    AlignedDatasetFull().preprocess_all()
    print("✅ batch_align() done")


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def pred(image_path: Path) -> None:
    """Predict the class of a single image."""
    print(Fore.MAGENTA + "⭐️ Predicting" + Style.RESET_ALL)
    from celebtwin.logic.preproc_face import NoFaceDetectedError
    from celebtwin.logic.registry import (
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


detector_choice = click.Choice([d.value for d in Detector])
ann_align_option = click.option(
    "-a", "--align", type=detector_choice, default=Detector.BUILTIN,
    help="""Detector backend for face alignment, defaults to built-in.""")

model_choice = click.Choice([m.value for m in Model])
ann_model_option = click.option(
    "-m", "--model", type=model_choice, default="Facenet",
    help="Model used to generate embeddings, defaults to Facenet.")

ann_backend_option = click.option(
    "-b", "--backend", type=click.Choice([b.value for b in ANNBackend]),
    default="annoy",
    help="ANNBackend used to build the index, defaults to annoy.")


def _make_strategy(backend: str, align: str, model: str) \
        -> 'logic.ann.ANNStrategy':
    """Make an ANN strategy from the command line options."""
    return ANNBackend(backend).strategy_class(
        Detector(align), Model(model))


@cli.command()
@ann_align_option
@ann_model_option
@ann_backend_option
@click.argument("validation_split", type=float)
def eval_ann(
        align: str, model: str, backend: str, validation_split: float) -> None:
    """Evaluate the ANN index."""
    print(Fore.BLUE + "Starting up" + Style.RESET_ALL)
    from celebtwin.logic.ann import ANNIndexEvaluator
    print(Fore.MAGENTA + "⭐️ Evaluating ANN index" + Style.RESET_ALL)
    strategy = _make_strategy(backend, align, model)
    ANNIndexEvaluator(strategy, validation_split).run()


@cli.command()
@ann_align_option
@ann_model_option
@ann_backend_option
def build_ann(align: str, model: str, backend: str) -> None:
    """Build an ANN index for the dataset."""
    print(Fore.BLUE + "Starting up" + Style.RESET_ALL)
    from celebtwin.logic.ann import ANNIndexBuilder
    print(Fore.MAGENTA + "⭐️ Building ANN index" + Style.RESET_ALL)
    strategy = _make_strategy(backend, align, model)
    ANNIndexBuilder(strategy).run()


@cli.command()
@ann_align_option
@ann_model_option
@ann_backend_option
@click.argument("image_path", type=click.Path(exists=True))
def pred_ann(
        image_path: Path, align: str, model: str, backend: str) -> None:
    """Predict the class of a single image using embedding proximity."""
    print(Fore.BLUE + "Starting up" + Style.RESET_ALL)
    from celebtwin.logic.ann import ANNReader
    from celebtwin.logic.preproc_face import NoFaceDetectedError
    print(Fore.MAGENTA + "⭐️ Predicting" + Style.RESET_ALL)
    strategy = ANNBackend(backend).strategy_class(
        Detector(align), Model(model))
    with ANNReader(strategy) as reader:
        try:
            class_, name = reader.find_image(image_path)
        except NoFaceDetectedError as error:
            print(Fore.RED + str(error) + Style.RESET_ALL)
            sys.exit(1)
    print(Fore.GREEN + f"Closest image: {class_}/{name}" + Style.RESET_ALL)


@cli.command()
@ann_align_option
@click.argument("image_path", type=click.Path(exists=True))
def align(image_path: Path, align: str) -> None:
    """Align a single image."""
    from celebtwin.logic.preproc_face import detect_faces
    assert align == "builtin"
    faces = detect_faces(image_path)
    print("Detected faces:")
    for i, face in enumerate(faces):
        print(f"Face {i+1}:")
        print(f"  Left eye: {face.left_eye}")
        print(f"  Right eye: {face.right_eye}")
        print(f"  Box: {face.box}")


def main():
    if not os.getenv('DEBUG'):
        cli()
    else:
        try:
            cli()
        except Exception:
            import traceback
            import ipdb  # type: ignore
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(exc_traceback)
