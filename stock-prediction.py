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
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500])
    plt.title(title)
    plt.show()

def separate_data(df: pd.DataFrame, num: int):
    first = df.head(len(df) - num)
    last = df.tail(num)
    return first, last

def prepare_data(df: pd.DataFrame):
    x = df.iloc[:, 0:1]
    y = df.iloc[:, 1:2]
    return x, y
    

input = load_dataset("input")
#show_dataframe_chart(input, "Input")
train_set, test_set = separate_data(input, 50)
#show_dataframe_info(x_train, "Train data")
#show_dataframe_info(x_test, "Test data")

x_train, y_train = prepare_data(train_set)
x_test, y_test = prepare_data(test_set)