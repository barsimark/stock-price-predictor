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
    print(df.head())
    print("Length: ", len(df))
    return df

def show_dataframe(df: pd.DataFrame):
    plt.figure(figsize = (9,5))
    plt.plot(range(df.shape[0]),(df['Open']))
    plt.xlabel("Days", fontsize=18)
    plt.ylabel("Open price", fontsize=18)
    plt.show()
    

input = load_dataset("input")
show_dataframe(input)