import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

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

def show_dataframe_info(df: pd.DataFrame, title: str):
    print("Title:", title)
    print(df.head())
    print("Length: ", len(df))

def show_dataframe_chart(df: pd.DataFrame, title: str):
    plt.figure(figsize = (9,5))
    plt.plot(range(df.shape[0]),(df['Open']))
    plt.xlabel("Days", fontsize=18)
    plt.ylabel("Open price", fontsize=18)
    plt.title(title)
    plt.show()

def separate_data(df: pd.DataFrame):
    last = df.tail(50)
    df.drop(df.tail(50).index, inplace=True)
    return df, last
    

input = load_dataset("input")
x_train, x_test = separate_data(input)
show_dataframe_info(x_train, "Train data")
show_dataframe_info(x_test, "Test data")
show_dataframe_chart(x_train, "Train data")
show_dataframe_chart(x_test, "Test data")