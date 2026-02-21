
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D, LeakyReLU
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, learning_rate=1e-3):
    model = Sequential([
        Conv1D(128, 5, padding='same', input_shape=input_shape),
        LeakyReLU(negative_slope=0.3),
        BatchNormalization(),
        MaxPooling1D(),
        Dropout(0.0),
        Conv1D(128, 5, padding='same'),
        LeakyReLU(negative_slope=0.3),
        BatchNormalization(),
        MaxPooling1D(),
        Dropout(0.0),
        LSTM(128),
        LeakyReLU(negative_slope=0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.0),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
