import pickle
import numpy as np
from Constants import *


# Function to take the data from the data files of CIFAR.
# Extracted from: https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


# Function to extract data from the dictionary to numpy arays
def get_data(file):
    dict_data = unpickle(file)
    X = np.asarray(dict_data[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict_data[b'labels'])
    Y = np.zeros((10, 10000))
    for i in range(10000):
        Y[Yraw[i], i] = 1
    return X, Y


# Main execution
if __name__ == "__main__":
    X1, Y1 = get_data(BATCH1)
    X2, Y2 = get_data(BATCH2)
    X3, Y3 = get_data(BATCH3)
    X4, Y4 = get_data(BATCH4)
    X5, Y5 = get_data(BATCH5)
    X_train = [X1, X2, X3, X4, X5]
    Y_train = [Y1, Y2, Y3, Y4, Y5]
    X_test, Y_test = get_data(TEST_BATCH)
