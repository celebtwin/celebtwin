"""Experiment management module.

Experiments combine datasets, models, and training configurations to run a
complete training and evaluation cycle.
"""

import time

import numpy as np
from celebtwin.ml_logic.data import Dataset
from celebtwin.ml_logic.model import Model
from celebtwin.ml_logic.registry import save_metadata


class Experiment:
    """Class representing an experiment.

    An experiment combines a dataset, a model, and training configurations to
    run a complete training and evaluation cycle.
    """

    def __init__(self, dataset: Dataset, model: Model, learning_rate: float,
                 patience: int):
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
        save_metadata(identifier, metadata)
