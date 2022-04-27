import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

class BaseLSTM():
    def __init__(self) -> None:
        self.model = Sequential()

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

class BasicModel(BaseLSTM):
    def __init__(self, sequence_length) -> None:
        super().__init__()
        self.model.add(LSTM(256, return_sequences=True, input_shape = (sequence_length, 1)))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(64))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.compile(optimizer='RMSprop', loss='mae')
        self.model.summary()

class ComplexModel(BaseLSTM):
    def __init__(self, sequence_length) -> None:
        super().__init__()
        self.model.add(LSTM(128, return_sequences=True, input_shape = (sequence_length, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))
        self.model.compile(optimizer='RMSprop', loss='mae')
        self.model.summary()
