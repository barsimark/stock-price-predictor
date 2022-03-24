# Stock price prediction

## Introduction

The aim of this project is to use deep learning to predict future prices of stocks. The repository is constantly evolving, as I try out various different approaches, and technologies. I am creating multiple models for different stocks, and indexes. Predictions will be made for individual shares, as well as utilizing multiple datasets to give more accurate price forecasts.

### Technologies

- Python 3.10
- Tensorflow 2.8

## Datasets

Nvidia and QQQ (US-based tech index) stock prices between January 2015 and February 2022 are used as an example.

The datasets are publicly available at MarketWatch in a downloadable .csv format in yearly chanks. 

![Nvidia prices](https://github.com/barsimark/stock-price-predictor/blob/master/images/Nvidia-prices.png)

![QQQ prices](https://github.com/barsimark/stock-price-predictor/blob/master/images/QQQ-prices.png)

## Models

### Basic LSTM model

Simple LSTM-based model to predict the prices of Nvidia stocks. This model will be used as a baseline for future models.

- Input: sequnce of stock price data
- Hidden layers: multiple LSTM, and Dropout layers
- Output: predicted stock price for the next day

As it can be seen on the performance chart, this model cannot be used in free running mode, meaning that the prediction only makes sense for the next day.

Performance on the test dataset:

![Basic model performance](https://github.com/barsimark/stock-price-predictor/blob/master/images/Basic-model-prediction.png)

![Basic model loss](https://github.com/barsimark/stock-price-predictor/blob/master/images/Basic-model-loss.png)

![Basic model regression](https://github.com/barsimark/stock-price-predictor/blob/master/images/Basic-model-regression-plot.png)

## Plans for the future

- Add new model with Nvidia and QQQ as inputs, and Nvidia prediction as output
- Add new model utilizing other RNN-based approaches
- Compare the results of each model and determine the best
- Give predictions for the future and use it in real trading

## Author

Mark Barsi