from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    ELU,
    Activation,
    Flatten,
    Dropout,
    Dense,
)
from typing import Callable


class EmotionVGGNet:
    @staticmethod
    def build(width: int, height: int, depth: int, classes: int):
        # initialise the model along with the input shape to be "channels last" and the channels dimension itself
        input_shape: tuple[int, int, int]
        chan_dim: int
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        # Block #1: first CONV => RELU => CONV => RELU => POOL
        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                kernel_initializer="he_normal",
                input_shape=input_shape,
            )
        )
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: third CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 4: first set of FC => RELU
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block 5: second set of FC => RELU
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block 6: softmax classifier
        model.add(Dense(classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))

        return model
