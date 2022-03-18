import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Show dataframe head and shape
def show_dataframe_info(data: np.array, title: str = "untitled"):
    print("Title:", title)
    print(data[:10])
    print("Shape: ", data.shape)
    print("")

## Show chart of stock price
def show_dataframe_chart(df: pd.DataFrame, title: str = ""):
    plt.figure(figsize = (9,5))
    plt.plot(range(df.shape[0]),(df['Open']))
    plt.xlabel("Days", fontsize=18)
    plt.ylabel("Open price", fontsize=18)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500])
    plt.title(title)
    plt.show()

def show_np_arrays(arr1: np.array, arr2: np.array, label1: str = "", label2: str = "", 
title: str = "", xlabel: str = "Days", ylabel: str = "Open price"):
    plt.figure(figsize = (9,5))
    plt.plot(arr1, label=label1)
    plt.plot(arr2, label=label2)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title)
    plt.legend(loc = "upper left")
    plt.show()

def show_np_array(arr: np.array, title: str = ""):
    plt.figure(figsize = (9,5))
    plt.plot(arr)
    plt.title(title)
    plt.show()

def show_regression_plot(arr1: np.array, arr2: np.array):
    sns.regplot(x=arr1, y=arr2)