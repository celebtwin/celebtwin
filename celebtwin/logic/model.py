from pathlib import Path

import keras  # type: ignore
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from keras import Input, Sequential, layers, optimizers
from keras.callbacks import EarlyStopping  # type: ignore

from celebtwin.logic import registry


class Model:
    """Machine learning model used for training and evaluation."""

    _model: keras.Model | None
    """The underlying Keras model."""

    _patience: int | None
    _learning_rate: float | None

    def __init__(self):
        self._model = None
        self._patience = None
        self._learning_rate = None

    def load(self, params: dict, keras_path: str | Path) -> None:
        """Load the model from a Keras file."""
        self._model = keras.models.load_model(keras_path)
        self._learning_rate = params['learning_rate']
        self._patience = params['patience']
        print("✅ Model loaded")

    @property
    def identifier(self) -> str:
        """Unique identifier for the model."""
        raise NotImplementedError("Implement identifier in a subclass.")

    def params(self) -> dict:
        """Return model parameters."""
        return {
            "model_identifier": self.identifier,
            "model_class": self.__class__.__name__,
            "patience": self._patience,
            "learning_rate": self._learning_rate,
        }

    def compile(self, learning_rate: float) -> None:
        """Compile the model."""
        raise NotImplementedError("Implement compile in a subclass.")

    def train(
            self, train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset, patience: int) -> dict:
        """Train the model on the provided datasets, return training history.
        """
        if self._model is None:
            raise ValueError("Model has not been built or loaded yet.")
        print(Fore.BLUE + "👟 Training model..." + Style.RESET_ALL)
        self._patience = patience
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
            verbose=1)
        history = self._model.fit(
            train_dataset, validation_data=validation_dataset,
            epochs=100000, callbacks=[early_stopping])
        train_len = len(train_dataset.file_paths)  # type: ignore
        val_accuracy = round(np.max(history.history['val_accuracy']), 3)
        print(f"✅ Model trained on {train_len} images with validation"
              f" accuracy: {val_accuracy}")
        return history.history

    def _model_path(self) -> str:
        """Path to the model file."""
        return f"models/{self.identifier}.keras"

    def save(self, identifier: str) -> None:
        """Save the model to the registry."""
        if self._model is None:
            raise ValueError("Model has not been built or loaded yet.")
        registry.save_model(self._model, identifier)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class probabilities for the input data."""
        if self._model is None:
            raise ValueError("Model has not been built or loaded yet.")
        X = np.expand_dims(X, axis=0)
        pred_probas = self._model.predict([X])
        assert len(pred_probas) == 1
        return pred_probas[0]


def load_model(params: dict, keras_path: str | Path) -> Model:

    class_map = {
        'SimpleLeNetModel': SimpleLeNetModel,
        'WeekendModel': WeekendModel}
    assert params['model_class'] in class_map, \
        f"Unexpected dataset class: {params['model_class']}"
    model = class_map[params['model_class']]()
    model.load(params, keras_path)
    return model


class SimpleLeNetModel(Model):
    """A simple LeNet-like model for image classification.

    This model is a baseline architecture similar to the one used for MNIST.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier for the model."""
        return '-'.join([
            "lenetv1",
            f"r{self._learning_rate}",
            f"p{self._patience}"])

    def build(self, input_shape: tuple, class_nb: int):
        super().__init__()

        self._model = Sequential([
            Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),  # Normalize pixel values

            # First Convolution & MaxPooling
            layers.Conv2D(8, (4, 4), padding='same', activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),

            # Second Convolution & MaxPooling
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),

            # Flattening
            layers.Flatten(),

            # One Fully Connected layer
            layers.Dense(10, activation='relu'),

            # Classification Layer: n outputs corresponding to n celebrities
            layers.Dense(class_nb, activation='softmax')])

    def compile(self, learning_rate: float) -> None:
        if self._model is None:
            raise ValueError("Model has not been built yet.")
        self._learning_rate = learning_rate
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,  # type: ignore
            metrics=['accuracy'])
        print("✅ Model compiled")


class WeekendModel(Model):
    """A more advanced model for image classification.

    This model is a VGG-like architecture.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier for the model."""
        return '-'.join([
            "v1weekend",
            f"r{self._learning_rate}",
            f"p{self._patience}"])

    def build(self, input_shape: tuple, class_nb: int):
        super().__init__()

        self._model = Sequential([
            Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),  # Normalize pixel values

            # BLOCK 1
            layers.Conv2D(20, (2, 2), padding='same', activation="relu"),
            layers.Conv2D(20, (3, 3), padding='same', activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            # BLOCK 2
            layers.Conv2D(38, (2, 2), padding='same', activation="relu"),
            layers.Conv2D(38, (3, 3), padding='same', activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),

            # BLOCK 3
            layers.Conv2D(76, (2, 2), padding='same', activation="relu"),
            layers.Conv2D(76, (2, 2), padding='same', activation="relu"),
            layers.Conv2D(76, (3, 3), padding='same', activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),

            # BLOCK 4
            layers.Conv2D(154, (2, 2), padding='same', activation="relu"),
            layers.Conv2D(154, (2, 2), padding='same', activation="relu"),
            layers.Conv2D(154, (3, 3), padding='same', activation="relu"),
            layers.MaxPool2D(pool_size=(2, 2)),

            # Flattening
            layers.Flatten(),

            # 2 Fully Connected layers - "Fully Connected" is equivalent to
            # saying "Dense"
            layers.Dense(1000, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(500, activation='relu'),

            # Classification Layer: n outputs corresponding to n celebrities
            layers.Dense(class_nb, activation='softmax')])

    def compile(self, learning_rate: float) -> None:
        if self._model is None:
            raise ValueError("Model has not been built yet.")
        self._learning_rate = learning_rate
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,  # type: ignore
            metrics=['accuracy'])
        print("✅ Model compiled")
