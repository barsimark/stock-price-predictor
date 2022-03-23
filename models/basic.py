import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

class BasicModel():
    def __init__(self, sequence_length) -> None:
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, input_shape = (sequence_length, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def train(self, x_train: np.array, y_train: np.array, epochs: int = 300, batch: int = 32):
        es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        self.history = self.model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch, callbacks=[es])

    def predict(self, dataset: np.array) -> np.array:
        return np.array(self.model.predict(dataset))

    def evaluate(self, x_test: np.array, y_test: np.array):
        print("Evaluating model")
        self.model.evaluate(x_test, y_test)

    def get_losses(self):
        return np.array(self.history.history['loss']), np.array(self.history.history['val_loss'])

    def free_running_prediction(self, dataset: np.array, length: int) -> np.array:
        future = []
        data = np.copy(dataset)
        for _ in range(length):
            result = self.predict(np.reshape(data, (1, -1, 1)))
            data = np.delete(data, (0), axis=0)
            data = np.append(data, result, axis=0)
            future.append(result)
            print(result)
        return np.reshape(np.array(future), (-1, 1))
