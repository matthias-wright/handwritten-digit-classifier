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


def forward_propagation(X, parameters, keep_prob):
    """
    Implements forward propagation for the neural network.
    :param X: data
    :param parameters: weights and bias units
    :return: A_out: neural network output, caches: list of caches for backprop
    """
    A = X
    D = np.ones((X.shape[0], X.shape[1])) # so that dropout is not applied to the input
    L = len(parameters) // 2
    caches = []

    for l in range(1, L):
        A_prev = A
        D_prev = D
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = utils.relu(Z)
        # inverted dropout
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
        A = np.multiply(A, D)
        A = np.divide(A, keep_prob)
        cache = (A_prev, D_prev, W, b, Z)
        caches.append(cache)

    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z_out = np.dot(W, A) + b
    A_out = utils.softmax(Z_out)
    cache = (A, D, W, b, Z_out)
    caches.append(cache)
    return A_out, caches


def back_propagation(A_out, Y, caches, keep_prob):
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
    A_prev, D, W, b, Z = caches[L-1]
    dZ = A_out - Y
    gradients['dW' + str(L)] = np.divide(np.dot(dZ, A_prev.T), m)
    gradients['db' + str(L)] = np.divide(np.sum(dZ, axis=1, keepdims=True), m)
    gradients['dA' + str(L - 1)] = np.dot(W.T, dZ)
    # dropout
    gradients['dA' + str(L - 1)] = np.multiply(gradients['dA' + str(L - 1)], D)
    gradients['dA' + str(L - 1)] = np.divide(gradients['dA' + str(L - 1)], keep_prob)

    for l in reversed(range(L-1)):
        A_prev, D, W, b, Z = caches[l]
        dZ = np.array(gradients['dA' + str(l + 1)], copy=True)
        dZ[Z <= 0] = 0
        gradients['dA' + str(l)] = np.dot(W.T, dZ)
        # dropout
        gradients['dA' + str(l)] = np.multiply(gradients['dA' + str(l)], D)
        gradients['dA' + str(l)] = np.divide(gradients['dA' + str(l)], keep_prob)

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


def init_adam(parameters):
    """
    Initializes v and s for the the Adam optimization algorithm.
    :param parameters: weights and bias units
    :return: v: dictionary for the moving averages of the gradients,
             s: dictionary for the moving averages of the squared gradients
    """
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        v['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))
        s['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        s['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))

    return v, s


def update_parameters_adam(parameters, gradients, v, s, t, beta1, beta2, learning_rate):
    """
    Performs a gradient descent update according to the Adam optimization algorithm.
    :param parameters: weights and bias units
    :param gradients: partial derivatives w.r.t. the weights and the bias units
    :param v: moving average of the gradients
    :param s: moving average of the squared gradient (RMSprop)
    :param beta1: exponential decay hyperparameter for v
    :param beta2: exponential decay hyperparameter for s
    :param learning_rate:
    :return:
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * gradients['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * gradients['db' + str(l)]
        v_bias_corr_W = v['dW' + str(l)] / (1 - beta1**t)
        v_bias_corr_b = v['db' + str(l)] / (1 - beta1**t)

        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * gradients['dW' + str(l)]**2
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * gradients['db' + str(l)]**2
        s_bias_corr_W = s['dW' + str(l)] / (1 - beta2**t)
        s_bias_corr_b = s['db' + str(l)] / (1 - beta2**t)

        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * np.divide(v_bias_corr_W, np.sqrt(s_bias_corr_W) + 1e-8)
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * np.divide(v_bias_corr_b, np.sqrt(s_bias_corr_b) + 1e-8)

        return parameters, v, s


def gradient_check(X, Y, parameters, gradients, epsilon):
    """
    Implements gradient checking in order to validate the backprop routine.
    :param X: data (single example)
    :param Y: true labels
    :param parameters: weights and bias units
    :param gradients: partial derivatives of the weights and bias units
    :param epsilon: the shift to the input to approximate the gradient
    :return: delta: the difference between the approximation and the computed gradient
    """
    parameters_vector, shapes = utils.roll_out_dict(parameters)
    gradient, _ = utils.roll_out_dict(gradients)
    n = parameters_vector.shape[0]
    Cost_plus = np.zeros((n, 1))
    Cost_minus = np.zeros((n, 1))
    approx_gradient = np.zeros((n, 1))

    for i in range(n):
        theta_plus = np.copy(parameters_vector)
        theta_plus[i][0] = theta_plus[i][0] + epsilon
        A_out, _ = forward_propagation(X, utils.roll_in_vector(theta_plus, shapes))
        Cost_plus[i] = utils.cross_entropy(A_out, Y)

        theta_minus = np.copy(parameters_vector)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        A_out, _ = forward_propagation(X, utils.roll_in_vector(theta_minus, shapes))
        Cost_minus[i] = utils.cross_entropy(A_out, Y)

        approx_gradient[i] = (Cost_plus[i] - Cost_minus[i]) / float(2 * epsilon)

    numerator = np.linalg.norm(gradient - approx_gradient)
    denominator = np.linalg.norm(gradient) + np.linalg.norm(approx_gradient)
    delta = numerator/float(denominator)
    return delta


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
    A_out, _ = forward_propagation(X, parameters, keep_prob=1)
    for i in range(m):
        if np.argmax(A_out[:, i]) == np.argmax(Y[:, i]):
            tp = tp + 1
    accuracy = tp / m
    return accuracy


def model(X_train, Y_train, X_test, Y_test, layers, learning_rate, beta1, beta2, epochs, mb_size, keep_prob):
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
    v, s = init_adam(parameters)
    t = 1
    mini_batches = get_mini_batches(X_train, Y_train, mb_size)
    costs = []

    for i in range(epochs):
        for j in range(len(mini_batches)):
            X_mb, Y_mb = mini_batches[j]
            A_out, caches = forward_propagation(X_mb, parameters, keep_prob)
            cost = utils.cross_entropy(A_out, Y_mb)
            gradients = back_propagation(A_out, Y_mb, caches, keep_prob)
            parameters, v, s = update_parameters_adam(parameters, gradients, v, s, t, beta1, beta2, learning_rate)
            #parameters = update_parameters(parameters, gradients, learning_rate)
            t = t + 1

        print('Epoch ' + str(i) + ', ' + 'cost: ' + str(cost))
        costs.append(cost)

    accuracy = test(X_test, Y_test, parameters)
    print('Accuracy: ' + str(accuracy))


