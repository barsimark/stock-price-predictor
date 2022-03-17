# Stock price prediction

## Introduction

The aim of this project is to use deep learning to predict future prices of stocks.
The repository is constantly evolving, as I try out various different approaches, and technologies.
I am creating multiple models for different stocks, and indexes.
Predictions will be made for individual shares, as well as utilizing multiple datasets to give more accurate price forecasts.

### Technologies

- Python 3.10
- Tensorflow 2.8

## Datasets

Nvidia and QQQ (US-based tech index) stock prices between January 2015 and February 2022 are used as an example.

The datasets are publicly available at MarketWatch in a downloadable .csv format in yearly chanks. 

![Nvidia prices](https://github.com/barsimark/stock-price-predictor/blob/master/images/Nvidia-prices.png)

![QQQ prices](https://github.com/barsimark/stock-price-predictor/blob/master/images/QQQ-prices.png)

## Models

### Basic model

Currently, there is only a single model available, although I am constantly working on it to improve its performance.

- Input: sequnce of stock price data
- Hidden layers: multiple LSTM, and Dropout layers
- Output: predicted stock price for the next day

Performance on the test dataset:

![Basic model performance](https://github.com/barsimark/stock-price-predictor/blob/master/images/Basic-model-prediction.png)

![Basic model loss](https://github.com/barsimark/stock-price-predictor/blob/master/images/Basic-model-loss.png)

## Plans for the future

- Improve the basic model to get more accurate results
- Add new model with Nvidia and QQQ as inputs, and Nvidia prediction as output
- Compare the results of each model and determine the best
- Reduce or eliminate outliers with using something like moving average
- Give predictions for the future and use it in real trading

## Author

Mark Barsi