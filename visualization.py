import pandas as pd
import matplotlib.pyplot as plt

## Show dataframe head and shape
def show_dataframe_info(df: pd.DataFrame, title: str):
    print("Title:", title)
    print(df.head())
    print("Length: ", df.shape)
    print("")

## Show chart of stock price
def show_dataframe_chart(df: pd.DataFrame, title: str):
    plt.figure(figsize = (9,5))
    plt.plot(range(df.shape[0]),(df['Open']))
    plt.xlabel("Days", fontsize=18)
    plt.ylabel("Open price", fontsize=18)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500])
    plt.title(title)
    plt.show()