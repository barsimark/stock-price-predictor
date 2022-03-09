import tensorflow as tf
import pandas as pd
import numpy as np
import os

import visualization as vs

def get_info():
    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_dataset(dir: str) -> pd.DataFrame:
    df = []
    for file in os.listdir(dir):
        df.append(pd.read_csv(dir + "/" + file))
    df = pd.concat(df)
    df['Date'] = pd.to_datetime(df.Date)
    df.sort_values(by='Date', inplace=True)
    return df

def separate_data(df: pd.DataFrame, num: int):
    first = df.head(len(df) - num)
    last = df.tail(num)
    return first, last

def prepare_data(df: pd.DataFrame):
    x = np.array(df.iloc[:, 0:1].values)
    y = np.array(df.iloc[:, 1:2].values)
    return x, y
    

input = load_dataset("input")
vs.show_dataframe_chart(input, "Input")
train_set, test_set = separate_data(input, int(len(input) * 0.1))

x_train, y_train = prepare_data(train_set)
x_test, y_test = prepare_data(test_set)
vs.show_dataframe_info(x_train, "x_train")
vs.show_dataframe_info(y_train, "y_train")
vs.show_dataframe_info(x_test, "x_test")
vs.show_dataframe_info(y_test, "y_test")