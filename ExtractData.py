import pickle
import numpy as np
# Function to take the data from the data files of CIFAR.
# Extracted from: https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


# Function to extract data from the dictionary to numpy arays
# https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib
def get_data(file):
    dict_data = unpickle(file)
    X = np.asarray(dict_data[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict_data[b'labels'])
    Y = np.zeros((10, 10000))
    for i in range(10000):
        Y[Yraw[i], i] = 1
    return X, Y
