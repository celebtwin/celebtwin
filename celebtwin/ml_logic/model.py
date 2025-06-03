import time
from typing import Tuple

import numpy as np
from colorama import Fore, Style

print(Fore.BLUE + "Loading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()
import tensorflow

end = time.perf_counter()
print(f"✅ TensorFlow loaded ({round(end - start, 2)}s)")

from keras import Input, Model, Sequential, layers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.data import Dataset


def initialize_model(input_shape: tuple, class_nb: int, colors: bool = True) -> Model:
    """Initialize the Neural Network with random weights."""
    # 3 or 1 depending on 'colors' value (True or False)
    nb_channels = 3 if colors else 1

    ####### Very very baseline model (similar to MNIST architecture)
    model = Sequential()

    model.add(Input(shape=(*input_shape, nb_channels)))

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (4, 4), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with n outputs corresponding to n celebrities
    model.add(layers.Dense(class_nb, activation='softmax'))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.001) -> Model:
    """Compile the Neural Network."""

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,  # type: ignore
                  metrics = ['accuracy'])

    print("✅ Model compiled")

    return model


def train_model(
        model: Model,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        patience: int,
    ) -> Tuple[Model, dict]:
    """Fit the model and return a tuple (fitted_model, history)."""

    print(Fore.BLUE + "👟 Training model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=[es],
        verbose=1  # type: ignore
    )

    # I guess cardinality should give the number of images, but it does not.
    # train_len = train_dataset.cardinality().numpy().item()
    train_len = len(train_dataset.file_paths)
    min_val_accuracy = round(np.min(history.history['val_accuracy']), 2)
    print(f"✅ Model trained on {train_len} images with min validation"
          f" accuracy: {min_val_accuracy}")

    return model, history.history
