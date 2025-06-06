import sys

import click
from celebtwin.ml_logic.preproc_face import preprocess_face_aligned
from colorama import Fore, Style


@click.group()
def cli():
    pass


@cli.command()
def train() -> None:
    """Train on a local dataset.

    Save validation metrics and the trained model.
    """
    print(Fore.MAGENTA + "⭐️ Training" + Style.RESET_ALL)
    from celebtwin.ml_logic.data import ColorMode, ResizeMode, SimpleDataset
    from celebtwin.ml_logic.experiment import Experiment
    from celebtwin.ml_logic.model import SimpleLeNetModel

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
    model = SimpleLeNetModel()
    model.build(
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


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def pred(image_path) -> None:
    """Predict the class of a single image."""
    print(Fore.MAGENTA + "⭐️ Predicting" + Style.RESET_ALL)
    from celebtwin.ml_logic.registry import NoModelFoundError, load_latest_experiment
    try:
        experiment = load_latest_experiment()
    except NoModelFoundError as error:
        print(Fore.RED + str(error) + Style.RESET_ALL)
        print(Fore.YELLOW + "Please train a model first." + Style.RESET_ALL)
        sys.exit(1)
    pred_probas, class_name = experiment.predict(image_path)
    print(Fore.GREEN + f"Predicted class: {class_name}" + Style.RESET_ALL)
    print(Fore.BLUE + f"Predicted probabilities: {pred_probas}"
          + Style.RESET_ALL)


@cli.command()
def test_align():
    """Test align on a local dataset."""
    chem = "raw_data/105_classes_pins_dataset/pins_Alexandra Daddario/Alexandra Daddario7_409.jpg"

    result = preprocess_face_aligned(
        chem, required_size=None, show_info=False)  # ,required_size=(64,64)
    print(f"✅ résultat : {result.shape}")


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
