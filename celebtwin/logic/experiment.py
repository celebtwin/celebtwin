"""Experiment management module.

Experiments combine datasets, models, and training configurations to run a
complete training and evaluation cycle.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from celebtwin.logic import data, model, registry


class Experiment:
    """Class representing an experiment.

    An experiment combines a dataset, a model, and training configurations to
    run a complete training and evaluation cycle.
    """

    def __init__(
            self, dataset: 'data.Dataset', model: model.Model,
            learning_rate: float, patience: int):
        self._dataset = dataset
        self._model = model
        self._learning_rate = learning_rate
        self._patience = patience

    def run(self):
        """Run the experiment."""
        train_dataset, val_dataset = self._dataset.load()

        actual_num_classes = len(train_dataset.class_names)  # type: ignore
        assert actual_num_classes == self._dataset.num_classes

        self._model.compile(learning_rate=self._learning_rate)
        self._history = self._model.train(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            patience=self._patience)

    def save_results(self):
        """Save the results of the experiment."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        val_accuracy = np.max(self._history['val_accuracy'])
        identifier = '--'.join([
            timestamp, self._dataset.identifier, self._model.identifier,
            str(round(val_accuracy, 3))])
        self._model.save(identifier)
        metadata = {
            'timestamp': timestamp,
            'dataset': self._dataset.params(),
            'model': self._model.params(),
            'metrics': {
                'val_accuracy': val_accuracy,
                'val_loss': np.min(self._history['val_loss']),
                'train_accuracy': np.max(self._history['accuracy']),
                'train_loss': np.min(self._history['loss'])
            },
            'history': self._history,
        }
        registry.save_metadata(identifier, metadata)

    def predict(self, image_path: Path) -> tuple[np.ndarray, str]:
        """Predict the class of an image provided by its path.

        Returns the predicted class probabilities and the class name.
        """
        image = self._dataset.load_prediction(image_path)
        pred = self._model.predict(image)
        assert self._dataset.class_names is not None
        class_name = self._dataset.class_names[np.argmax(pred)]
        return pred, class_name


@lru_cache(maxsize=1)
def load_experiment(metadata_path: Path, model_path: Path) -> Experiment:
    """Load an experiment from the registry."""
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    dataset = data.load_dataset(metadata['dataset'])
    model_instance = model.load_model(metadata['model'], model_path)
    learning_rate = metadata['model']['learning_rate']
    patience = metadata['model']['patience']
    experiment = Experiment(dataset, model_instance, learning_rate, patience)
    experiment._history = metadata['history']
    return experiment
