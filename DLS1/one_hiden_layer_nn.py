#Necesaary imports
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

    