import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

import visualization as vs
from models.lstm import BasicModel

TEST_SET_RATIO = 0.1
TRAIN_SEQUENCE_LENGTH = 20

def load_dataset(dir: str) -> pd.DataFrame:
    df = []
    for file in os.listdir(dir):
        df.append(pd.read_csv(dir + "/" + file))
    df = pd.concat(df)
    df['Date'] = pd.to_datetime(df.Date)
    df.drop_duplicates(inplace=True)
    df.sort_values(by='Date', inplace=True)
    return df

def get_prices_from_dataframe(df: pd.DataFrame) -> np.array:
    return np.array(df.iloc[:, 1:2].values)

def separate_data(arr: np.array, num: int):
    first = arr[:num]
    last = arr[num:]
    return first, last

def create_x_y_matrices(train_set: np.array):
    x_train = []
    y_train = []
    for i in range(TRAIN_SEQUENCE_LENGTH, train_set.shape[0]):
        x_train.append(train_set[i-TRAIN_SEQUENCE_LENGTH:i, 0])
        y_train.append(train_set[i, 0])
    return np.array(x_train), np.array(y_train)

input = load_dataset("input/Nvidia")
input_prices = get_prices_from_dataframe(input)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = np.array(scaler.fit_transform(input_prices))
train_set, test_set = separate_data(scaled_prices, int(scaled_prices.shape[0] * (1 - TEST_SET_RATIO)))

x_train, y_train = create_x_y_matrices(train_set)
x_test, y_test = create_x_y_matrices(test_set)
x_train = np.reshape(x_train, (-1, TRAIN_SEQUENCE_LENGTH, 1))
x_test = np.reshape(x_test, (-1, TRAIN_SEQUENCE_LENGTH, 1))

model = BasicModel(TRAIN_SEQUENCE_LENGTH)
model.train(x_train, y_train)
model.evaluate(x_test, y_test)
predicted = model.predict(x_test)
future = model.free_running_prediction(x_test[0], x_test.shape[0])

vs.show_regression_plot(y_test, predicted)
y_test = np.reshape(y_test, (-1, 1))
y_test = scaler.inverse_transform(y_test)
predicted = scaler.inverse_transform(predicted)
future = scaler.inverse_transform(future)
loss, val_loss = model.get_losses()
vs.show_np_arrays([y_test, predicted, future], ["Actual price", "Predicted price", "Free running price"], "Nvidia price prediction")
vs.show_np_arrays([loss, val_loss], ["Training", "Validation"], "Model's loss", "Epoch", "Loss")