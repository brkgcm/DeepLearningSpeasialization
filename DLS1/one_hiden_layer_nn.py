#Necesaary imports
from functools import cache
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets

def layer_sizes(X, Y, hidden_layer_size = 4):
    n_x = X.shape[0]
    n_h = hidden_layer_size
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1':w1, 'b1':b1, 'W2':w2, 'b2':b2}
    return parameters

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagation(X, parameters):
    #extract parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np. tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'z1':Z1, 'a1':A1, 'z2':Z2, 'a2':A2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1] #number of examples
    cost = np.sum((Y*np.log(A2))+((1-Y)*np.log(1-A2)))/(-m)
    cost = np.squeeze(np.array(cost))
    return cost

    