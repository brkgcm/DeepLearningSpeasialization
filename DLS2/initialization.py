from DLS1 import l_layer_nn as lnn
import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters_zeros(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def initialize_parameters_random(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_parameters_he(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
    return parameters

