from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling1D,
    LeakyReLU,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, learning_rate=1e-3):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(128, 5, padding="same"),
        LeakyReLU(negative_slope=0.3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, 5, padding="same"),
        LeakyReLU(negative_slope=0.3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        LSTM(128),
        LeakyReLU(negative_slope=0.3),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
