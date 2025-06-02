import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import Tensor
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers, Input
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(input_shape: tuple, class_nb: int, colors: bool = True) -> Model:
    """
    Initialize the Neural Network with random weights
    """
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


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    ### Model compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics = ['accuracy'])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: Tensor,
        y: np.ndarray,
        batch_size=32,
        patience=5,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_accuracy']), 2)}")

    return model, history
