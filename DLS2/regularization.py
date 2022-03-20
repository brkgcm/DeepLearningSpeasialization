# imports
import numpy as np
import DLS1.l_layer_nn as lnn


# function to compute the cost with regularization
def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    weights = []
    L = len(parameters) // 2
    for l in range(1, L + 1):
        weights.append(parameters['W' + str(l)])

    cross_entropy_cost = lnn.compute_cost(AL, Y)
    L2_regularization_cost = np.sum([np.sum(np.square(w)) for w in weights]) * (lambd / (2 * m))
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    L = len(cache) // 2
    grads = {'dZ' + str(L): cache['A' + str(L)] - Y}
    grads['dW' + str(L)] = 1 / m * np.dot(grads['dZ' + str(L)], cache['A' + str(L - 1)].T) + (lambd / m) * cache[
        'W' + str(L)]
    grads['db' + str(L)] = 1 / m * np.sum(grads['dZ' + str(L)], axis=1, keepdims=True)
    for l in range(L - 1, 0, -1):
        grads['dA' + str(l)] = np.dot(cache['W' + str(l + 1).T, grads['dZ' + str(l + 1)]])
        grads['dZ' + str(l)] = np.multiply(grads['dA' + str(l)], np.int64(cache['A' + str(l)] > 0))
        if l == 1:
            grads['dW' + str(l)] = 1. / m * np.dot(grads['dZ' + str(l)], X.T) + (lambd / m) * cache['W' + str(l)]
        else:
            grads['dW' + str(l)] = 1. / m * np.dot(grads['dZ' + str(l)], cache['A' + str(l - 1)].T) + (lambd / m) * \
                                   cache['W' + str(l)]
        grads['db' + str(l)] = 1. / m * np.sum(grads['dZ' + str(l)], axis=1, keepdims=True)

    return grads
