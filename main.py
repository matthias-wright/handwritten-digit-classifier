# -*- coding: utf-8 -*-

import numpy as np
import model


def get_data(path):
    """
    Returns the data X and the corresponding labels Y. Y is a one-hot matrix.
    :param path: file path of the csv dataset
    :return: X and Y
    """
    data = np.loadtxt(path, delimiter=',')
    X = data[:, 1:].T
    Y_ = data[:, :1].reshape(-1).astype(int)
    Y = np.eye(10)[Y_].T
    return X, Y


X_train, Y_train = get_data('mnist_train.csv')
X_test, Y_test = get_data('mnist_test.csv')

model.model(X_train, Y_train, X_test, Y_test, layers=[X_train.shape[0], 500, 300, 100, 10], learning_rate=0.01, epochs=20, mb_size=128)