import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class BasicModel():
    def __init__(self, sequence_length) -> None:
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape = (sequence_length, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(60))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def train(self, x_train: np.array, y_train: np.array, epochs: int = 100, batch: int = 32):
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch)

    def predict(self, dataset: np.array) -> np.array:
        return np.array(self.model.predict(dataset))

    def get_losses(self):
        return np.array(self.history.history['loss'])