import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

import visualization as vs

TEST_SET_RATIO = 0.1
TRAIN_SEQUENCE_LENGTH = 20

def load_dataset(dir: str) -> pd.DataFrame:
    df = []
    for file in os.listdir(dir):
        df.append(pd.read_csv(dir + "/" + file))
    df = pd.concat(df)
    df['Date'] = pd.to_datetime(df.Date)
    df.sort_values(by='Date', inplace=True)
    return df

def get_prices_from_dataframe(df: pd.DataFrame) -> np.array:
    return np.array(df.iloc[:, 1:2].values)

def separate_data(arr: np.array, num: int):
    first = arr[num:]
    last = arr[:num]
    return first, last

def create_train_matrices(train_set: np.array):
    x_train = []
    y_train = []
    for i in range(TRAIN_SEQUENCE_LENGTH, train_set.shape[0]):
        x_train.append(train_set[i-TRAIN_SEQUENCE_LENGTH:i, 0])
        y_train.append(train_set[i, 0])
    return np.array(x_train), np.array(y_train)

input = load_dataset("input")
input_prices = get_prices_from_dataframe(input)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = np.array(scaler.fit_transform(input_prices))
train_set, test_set = separate_data(scaled_prices, int(scaled_prices.shape[0] * TEST_SET_RATIO))

x_train, y_train = create_train_matrices(scaled_prices)
x_train = np.reshape(x_train, (-1, TRAIN_SEQUENCE_LENGTH, 1))
vs.show_dataframe_info(x_train)
vs.show_dataframe_info(y_train)