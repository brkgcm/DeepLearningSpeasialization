#Packages
from turtle import circle
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


#sigmoid backward propagation
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


# relu backward propagation
def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

#linear part of the backward propagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


# backward propagation activation->linear
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation = 'relu':
        dZ = relu_backward(dA, cache)
        dA_prev, dW, db =  linear_backward(dZ, cache)
    if activation = 'sigmoid':
        dZ = sigmoid_backward(dA, cache)
        dA_prev, dW, db = linear_backward(dZ, cache)
    return dA_prev, dW, db


# L layer backward propagation
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1] # number of examples
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    #Gradient of the cost
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer of (SIGMOID -> LINEAR) gradients.
    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    grads['dA' + str(L-1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp

    #Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, activation='relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l+1)] = dW_temp
        grads['db' + str(l+1)] = db_temp
    
    return grads

# update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters['W'+str(l+1)] =parameters['W'+str(l+1)] - grads['dW' + str(l+1)]*learning_rate
        parameters['b'+str(l+1)] =parameters['b'+str(l+1)] - grads['db' + str(l+1)]*learning_rate
    return parameters

