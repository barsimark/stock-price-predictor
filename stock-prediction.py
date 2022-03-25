import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

import visualization as vs
from models.lstm import BasicModel
from models.esn import ESNModel

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

def get_moving_average(data:np.array, window:int) -> np.array:
    averages = []
    for i in range(window):
        averages.append(data[i])
    for i in range(data.shape[0] - window):
        current = data[i:i+window]
        averages.append(sum(current) / window)
    return np.array(averages)

def prediction_with_moving_average(data:np.array):
    average_5 = get_moving_average(data, 5)
    average_20 = get_moving_average(data, 20)
    average_50 = get_moving_average(data, 50)
    vs.show_np_arrays(
        [data[-160:], average_5[-160:], average_20[-160:], average_50[-160:]], 
        ["Price", "5-day moving average", "20-day moving average", "50-day moving average"], 
        "Moving average", 
        ylabel="Price")

def prediction_with_basic_lstm(x_train:np.array, y_train:np.array, x_test:np.array, y_test:np.array, scaler:MinMaxScaler):
    model = BasicModel(TRAIN_SEQUENCE_LENGTH)
    model.train(x_train, y_train, epochs=1)
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

def prediction_with_esn(x_train:np.array, y_train:np.array, x_test:np.array, y_test:np.array, scaled:np.array, scaler:MinMaxScaler):
    model = ESNModel(500)
    ESN_FUTURE = 2
    preds_short = model.train_and_predict(
        x_train.shape[0]//ESN_FUTURE*ESN_FUTURE, 
        ESN_FUTURE, 
        x_test.shape[0]//ESN_FUTURE*ESN_FUTURE,
        np.reshape(scaled[20:], (-1)),
        truth=y_test,
        offset=TRAIN_SEQUENCE_LENGTH
    )
    ESN_FUTURE = 5
    preds_long = model.train_and_predict(
        x_train.shape[0]//ESN_FUTURE*ESN_FUTURE, 
        ESN_FUTURE, 
        x_test.shape[0]//ESN_FUTURE*ESN_FUTURE,
        np.reshape(scaled[20:], (-1)),
        truth=y_test,
        offset=TRAIN_SEQUENCE_LENGTH
    )
    y_test = np.reshape(y_test, (-1, 1))
    preds_short = np.reshape(preds_short, (-1, 1))
    preds_long = np.reshape(preds_long, (-1, 1))
    y_test = scaler.inverse_transform(y_test)
    preds_short = scaler.inverse_transform(preds_short)
    preds_long = scaler.inverse_transform(preds_long)
    vs.show_np_arrays([y_test, preds_short, preds_long], ["Actual price", "Prediction for the next 2 days", "Prediction for the next 5 days"], "Nvidia price prediction")

input = load_dataset("input/Nvidia")
input_prices = get_prices_from_dataframe(input)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = np.array(scaler.fit_transform(input_prices))
train_set, test_set = separate_data(scaled_prices, int(scaled_prices.shape[0] * (1 - TEST_SET_RATIO)))

x_train, y_train = create_x_y_matrices(train_set)
x_test, y_test = create_x_y_matrices(test_set)
x_train = np.reshape(x_train, (-1, TRAIN_SEQUENCE_LENGTH, 1))
x_test = np.reshape(x_test, (-1, TRAIN_SEQUENCE_LENGTH, 1))

prediction_with_moving_average(input_prices)