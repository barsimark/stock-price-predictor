import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

import visualization as vs
from models.lstm import BasicModel, ComplexModel
from models.esn import ESNModel
import models.classic as cls

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

def create_x_y_matrices_complex(basic_set: np.array, baseline_set: np.array):
    x = np.zeros((len(basic_set) - TRAIN_SEQUENCE_LENGTH, TRAIN_SEQUENCE_LENGTH, 2))
    y = np.zeros((len(basic_set) - TRAIN_SEQUENCE_LENGTH, 1))
    for i in range(TRAIN_SEQUENCE_LENGTH, basic_set.shape[0]):
        for j in range(TRAIN_SEQUENCE_LENGTH):
            x[i-TRAIN_SEQUENCE_LENGTH][j][0] = basic_set[i-TRAIN_SEQUENCE_LENGTH+j, 0]
            x[i-TRAIN_SEQUENCE_LENGTH][j][1] = baseline_set[i-TRAIN_SEQUENCE_LENGTH+j, 0]
        y[i-TRAIN_SEQUENCE_LENGTH][0] = basic_set[i]
    return x, y

def prediction_with_moving_average(data:np.array, length:int):
    average_5 = cls.get_moving_average(data, 5)
    average_20 = cls.get_moving_average(data, 20)
    average_50 = cls.get_moving_average(data, 50)
    vs.show_np_arrays(
        [data[-length:], average_5[-length:], average_20[-length:], average_50[-length:]], 
        ["Price", "5-day moving average", "20-day moving average", "50-day moving average"], 
        "Moving average", 
        ylabel="Price"
    )

def prediction_with_interpolation(data:np.array, length:int, future:int):
    y1 = cls.iterative_interpolation(data, length, future, 1)
    y2 = cls.iterative_interpolation(data, length, future, 2)
    vs.show_np_arrays(
        [data[-length:], y1, y2],
        ["Actual price", "Linear extrapolation", "Quadratic extrapolation"],
        "Nvidia prices using " + str(future) + "-day extrapolation"
    )

def prediction_with_basic_lstm(x_train:np.array, y_train:np.array, x_test:np.array, y_test:np.array, scaler:MinMaxScaler):
    model = BasicModel(TRAIN_SEQUENCE_LENGTH)
    model.train(x_train, y_train, epochs=500)
    model.evaluate(x_test, y_test)
    predicted = model.predict(x_test)

    vs.show_regression_plot(y_test, predicted)

    y_test = np.reshape(y_test, (-1, 1))
    y_test = scaler.inverse_transform(y_test)
    predicted = scaler.inverse_transform(predicted)
    last = [item[-1] for item in x_test]
    last = np.reshape(last, (-1, 1))
    last = scaler.inverse_transform(last)
    vs.show_np_arrays(
        [y_test, predicted, last], 
        ["Actual price", "Predicted price", "Naiv prediction"], 
        "Nvidia price prediction"
    )

    loss, val_loss = model.get_losses()
    vs.show_np_arrays(
        [loss, val_loss], 
        ["Training", "Validation"], 
        "Model's loss", 
        "Epoch", 
        "Loss"
    )

def prediction_with_complex_lstm(x_train:np.array, y_train:np.array, x_test:np.array, y_test:np.array, scaler:MinMaxScaler):
    model = ComplexModel(TRAIN_SEQUENCE_LENGTH)
    model.train(x_train, y_train, epochs=500)
    model.evaluate(x_test, y_test)
    predicted = model.predict(x_test)

    vs.show_regression_plot(y_test, predicted)

    y_test = np.reshape(y_test, (-1, 1))
    y_test = scaler.inverse_transform(y_test)
    predicted = scaler.inverse_transform(predicted)
    last = [item[-1][0] for item in x_test]
    last = np.reshape(last, (-1, 1))
    last = scaler.inverse_transform(last)
    vs.show_np_arrays(
        [y_test, predicted, last], 
        ["Actual price", "Predicted price", "Naiv prediction"], 
        "Nvidia price prediction"
    )

    loss, val_loss = model.get_losses()
    vs.show_np_arrays(
        [loss, val_loss], 
        ["Training", "Validation"], 
        "Model's loss", 
        "Epoch", 
        "Loss"
    )

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
    vs.show_np_arrays(
        [y_test, preds_short, preds_long], 
        ["Actual price", "Prediction for the next 2 days", "Prediction for the next 5 days"], 
        "Nvidia price prediction"
    )

def get_simple_data():
    input = load_dataset("input/Nvidia")
    input_prices = get_prices_from_dataframe(input)
    train_set, test_set = separate_data(input_prices, int(input_prices.shape[0] * (1 - TEST_SET_RATIO)))
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_set = np.array(scaler.fit_transform(train_set))
    test_set = np.array(scaler.transform(test_set))

    x_train, y_train = create_x_y_matrices(train_set)
    x_test, y_test = create_x_y_matrices(test_set)
    x_train = np.reshape(x_train, (-1, TRAIN_SEQUENCE_LENGTH, 1))
    x_test = np.reshape(x_test, (-1, TRAIN_SEQUENCE_LENGTH, 1))

    return x_train, y_train, x_test, y_test, scaler

def get_complex_data():
    input = load_dataset("input/Nvidia")
    input_prices = get_prices_from_dataframe(input)
    train_set, test_set = separate_data(input_prices, int(input_prices.shape[0] * (1 - TEST_SET_RATIO)))

    input_prices_qqq = get_prices_from_dataframe(load_dataset('input/QQQ'))
    train_qqq, test_qqq = separate_data(input_prices_qqq, int(input_prices_qqq.shape[0] * (1 - TEST_SET_RATIO)))

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_set = np.array(scaler.fit_transform(train_set))
    test_set = np.array(scaler.transform(test_set))

    qqq_scaler = MinMaxScaler(feature_range=(0, 1))
    train_qqq = np.array(qqq_scaler.fit_transform(train_qqq))
    test_qqq = np.array(qqq_scaler.fit_transform(test_qqq))
    
    x_train, y_train = create_x_y_matrices_complex(train_set, train_qqq)
    x_test, y_test = create_x_y_matrices_complex(test_set, test_qqq)

    return x_train, y_train, x_test, y_test, scaler

x_train, y_train, x_test, y_test, scaler = get_complex_data()
prediction_with_complex_lstm(x_train, y_train, x_test, y_test, scaler)