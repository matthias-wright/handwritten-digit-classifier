# -*- coding: utf-8 -*-

import numpy as np


def relu(z):
    """
    Implements the ReLU function.
    :param z: input
    :return: ReLU(input)
    """
    return np.maximum(0, z)


def softmax(z):
    """
    Implements the softmax function.
    :param z: input
    :return: sigmoid(input)
    """
    t = np.exp(z)
    return np.divide(t, np.sum(t, axis=0))


def cross_entropy(A_out, Y):
    """
    Computes the cross entropy.
    :param A_out: neural network output
    :param Y: true labels
    :return: cross entropy cost
    """
    m = Y.shape[1]
    cost = - np.sum(np.sum(np.multiply(Y, np.log(A_out)), axis=0) + np.sum(np.multiply(1 - Y, np.log(1 - A_out)), axis=0)) / m
    return np.squeeze(cost)