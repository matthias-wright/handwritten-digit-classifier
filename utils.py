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
    cost = - np.sum(np.sum(np.multiply(Y, np.log(A_out + 1e-10)), axis=0) + np.sum(np.multiply(1 - Y, np.log(1 - A_out + 1e-10)), axis=0)) / m
    return np.squeeze(cost)


def roll_out_dict(dictionary):
    """
    Rolls out all of the matrices and vectors in the dictionary into a single vector.
    :param dictionary: dictionary of matrices and vectors
    :return: big_vector: single vector containing every element, shapes: dictionary containing the original the shapes
    """
    keys = list(dictionary.keys())
    shapes = {}
    count = 0
    for key in keys:
        if key.startswith('W') or key.startswith('b') or key.startswith('dW') or key.startswith('db'):
            shapes[key] = dictionary[key].shape
            vector = np.reshape(dictionary[key], (-1, 1))
            if count == 0:
                big_vector = vector
            else:
                big_vector = np.concatenate((big_vector, vector), axis=0)
            count = count + 1

    return big_vector, shapes


def roll_in_vector(vector, shapes):
    """
    Takes in a vector and reshapes
    :param vector: vector containing all of the elements
    :param shapes: dictionary containing the original the shapes
    :return: dictionary containing the original matrices and vectors
    """
    keys = list(shapes.keys())
    dictionary = {}
    start = 0
    end = 0
    for key in keys:
        num_elements = shapes[key][0] * shapes[key][1]
        end = end + num_elements
        dictionary[key] = vector[start:end, :].reshape(shapes[key])
        start = start + num_elements

    return dictionary
