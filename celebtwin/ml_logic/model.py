import keras
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from keras import Input, Sequential, layers, optimizers
from keras.callbacks import EarlyStopping
from celebtwin.ml_logic.registry import save_model


class Model:
    """Machine learning model used for training and evaluation."""

    _model: keras.Model
    """The underlying Keras model."""

    _patience: int

    @property
    def identifier(self) -> str:
        """Unique identifier for the model."""
        raise NotImplementedError("Implement identifier in a subclass.")

    def params(self) -> dict:
        """Return model parameters."""
        return {
            "model_identifier": self.identifier,
            "model_class": self.__class__.__name__,
            "patience": self._patience
        }

    def compile(self, learning_rate: float) -> None:
        """Compile the model."""
        raise NotImplementedError("Implement compile in a subclass.")

    def train(
            self, train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset, patience: int) -> dict:
        """Train the model on the provided datasets, return training history."""
        print(Fore.BLUE + "👟 Training model..." + Style.RESET_ALL)
        self._patience = patience
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
            verbose=1)
        history = self._model.fit(
            train_dataset, validation_data=validation_dataset,
            epochs=100, callbacks=[early_stopping])
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
        save_model(self._model, identifier)


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

    _learning_rate: float

    def __init__(self, input_shape: tuple, class_nb: int):
        super().__init__()

        self._model = Sequential([
            Input(shape=input_shape),

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

    def params(self) -> dict:
        params = super().params()
        params.update({
            "learning_rate": self._learning_rate
        })
        return params

    def compile(self, learning_rate: float) -> None:
        self._learning_rate = learning_rate
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,  # type: ignore
            metrics=['accuracy'])
        print("✅ Model compiled")
