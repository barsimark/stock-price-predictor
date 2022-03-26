import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def get_moving_average(data:np.array, window:int) -> np.array:
    averages = []
    for i in range(window):
        averages.append(data[i])
    for i in range(data.shape[0] - window):
        current = data[i:i+window]
        averages.append(sum(current) / window)
    return np.array(averages)

def interpolation(x:np.array, y:np.array, future:int, order:int=1):
    pred = np.arange(x.shape[0], x.shape[0] + future, 1)
    func = InterpolatedUnivariateSpline(x, y, k=order)
    return func(pred)

def iterative_interpolation(data:np.array, length:int, interpolation_length:int, order:int=1):
    data = np.reshape(data, (-1))
    train_length = data.shape[0] - length
    y = []
    for i in range(0, length, interpolation_length):
        y.append(interpolation(np.arange(train_length), data[i:i + train_length], interpolation_length, order))
    y = np.array(y)
    y = np.reshape(y, (-1))
    return y