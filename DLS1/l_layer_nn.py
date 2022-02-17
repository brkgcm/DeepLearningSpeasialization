#Packages
import numpy as np

#sigmoid functuion
def sigmoid(z):
    return 1/(1+np.exp(-z)), z

#relu function
def relu(z):
    return np.maximum(0, z), z

#Create and itialize the parameters
def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn{layer_dims[l], layer_dims[l-1]} * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters


#linear part of the forward propagation
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


#forward propagation linear->activation
def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    return A, (linear_cache, activation_cache)


#L layer forward porpagation
def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2 #number of layers in the neural network
    A_prev = X

    for l in range(1, L):
        A_prev, cache =  linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches


# cross-entropy cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    cost = np.squeeze(np.array(cost))
    return cost




    
