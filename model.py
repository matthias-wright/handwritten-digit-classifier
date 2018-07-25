# -*- coding: utf-8 -*-

"""
Handwritten digit classifier, trained on the MNIST dataset: http://yann.lecun.com/exdb/mnist/
"""

__author__ = 'Matthias Wright'

import numpy as np
import math
import utils


def init_parameters(layers):
    """
    Initializes the weights and bias units.
    :param layers: list containing the number of neurons per layer
    :return: parameter dictionary
    """
    parameters = {}
    L = len(layers)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers[l], 1))

    return parameters


def get_mini_batches(X, Y, mb_size):
    """
    This function returns a list containing the mini-batches.
    :param mb_size: size of the mini batch
    :return: list containing the mini-batches
    """
    i = np.random.permutation(X.shape[1])
    X = X[:, i]
    Y = Y[:, i]
    m = X.shape[1]
    num_batches = int(math.floor(m/mb_size))
    mini_batches = []
    for k in range(num_batches):
        mini_batch_X = X[:, k * mb_size:(k+1) * mb_size]
        mini_batch_Y = Y[:, k * mb_size:(k+1) * mb_size]
        mini_batches.append([mini_batch_X, mini_batch_Y])

    if m % mb_size != 0:
        mini_batch_X = X[:, -(m % mb_size):]
        mini_batch_Y = Y[:, -(m % mb_size):]
        mini_batches.append([mini_batch_X, mini_batch_Y])
    return mini_batches


def forward_propagation(X, parameters):
    """
    Implements forward propagation for the neural network.
    :param X: data
    :param parameters: weights and bias units
    :return: A_out: neural network output, caches: list of caches for backprop
    """
    A = X
    L = len(parameters) // 2
    caches = []

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = utils.relu(Z)
        cache = (A_prev, W, b, Z)
        caches.append(cache)

    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z_out = np.dot(W, A) + b
    A_out = utils.softmax(Z_out)
    cache = (A, W, b, Z_out)
    caches.append(cache)
    return A_out, caches


def back_propagation(A_out, Y, caches):
    """
    Implements the backprop routine for the neural network.
    :param A_out: neural network output
    :param Y: true labels
    :param caches: A, W, b, Z for every layer
    :return: partial derivative of W, A, and b w.r.t. the cost for every layer
    """
    gradients = {}
    L = len(caches)
    m = Y.shape[1]
    Y = Y.reshape(A_out.shape)
    # partial derivative of the cost function w.r.t. the output of the neural network
    A_prev, W, b, Z = caches[L-1]
    dZ = A_out - Y
    gradients['dW' + str(L)] = np.divide(np.dot(dZ, A_prev.T), m)
    gradients['db' + str(L)] = np.divide(np.sum(dZ, axis=1, keepdims=True), m)
    gradients['dA' + str(L - 1)] = np.dot(W.T, dZ)

    for l in reversed(range(L-1)):
        A_prev, W, b, Z = caches[l]
        dZ = np.array(gradients['dA' + str(l+1)], copy=True)
        dZ[Z <= 0] = 0
        gradients['dA' + str(l)] = np.dot(W.T, dZ)
        gradients['dW' + str(l + 1)] = np.divide(np.dot(dZ, A_prev.T), m)
        gradients['db' + str(l + 1)] = np.divide(np.sum(dZ, axis=1, keepdims=True), m)

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    """
    Performs a gradient descent update.
    :param parameters: weights and bias units
    :param gradients: partial derivatives w.r.t. the weights and the bias units
    :param learning_rate: gradient descent step size
    :return: updated gradients
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * gradients['db' + str(l)]

    return parameters


def test(X, Y, parameters):
    """
    Tests the accuracy of the classifier.
    :param X: test data
    :param Y: true labels
    :param parameters: weight and bias units
    :return: accuracy
    """
    m = X.shape[1]
    tp = 0
    A_out, _ = forward_propagation(X, parameters)
    for i in range(m):
        if np.argmax(A_out[:, i]) == np.argmax(Y[:, i]):
            tp = tp + 1
    accuracy = tp / m
    return accuracy


def model(X_train, Y_train, X_test, Y_test, layers, learning_rate, epochs, mb_size):
    """
    Implements a neural network.
    :param X: numpy array of shape (num_features, num_examples)
    :param Y: labels
    :param layers: list containing the number of neurons in each layer
    :param learning_rate: learning rate for the gradient update
    :param epochs: number of epochs
    """
    assert X_train.shape[0] == layers[0]

    parameters = init_parameters(layers)
    mini_batches = get_mini_batches(X_train, Y_train, mb_size)
    costs = []

    for i in range(epochs):
        for j in range(len(mini_batches)):
            X_mb, Y_mb = mini_batches[j]
            A_out, caches = forward_propagation(X_mb, parameters)
            cost = utils.cross_entropy(A_out, Y_mb)
            gradients = back_propagation(A_out, Y_mb, caches)
            update_parameters(parameters, gradients, learning_rate)

        print('Epoch ' + str(i) + ', ' + 'cost: ' + str(cost))
        costs.append(cost)

    accuracy = test(X_test, Y_test, parameters)
    print('Accuracy: ' + str(accuracy))


