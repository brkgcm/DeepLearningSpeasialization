from DLS1 import l_layer_nn as lnn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds


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

def model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, initialization="he"):
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    grads = {}
    costs = []
    m = X.shape[1] # number of examples

    for i in range(0, num_iterations):
        # Forward propagation
        AL, caches = lnn.L_model_forward(X, parameters)

        # Compute cost
        cost = lnn.compute_cost(AL, Y)

        # Backward propagation
        grads = lnn.L_model_backward(AL, Y, caches)

        if print_cost  and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

        # Update parameters
        parameters = lnn.update_parameters(parameters, grads, learning_rate)

    # Plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# make a synthetic dataset
X, y = ds.make_circles(n_samples=10000,factor=0.5, random_state=0, noise=0.07)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor="k")
plt.show()
X = X.T

layers_dims = [X.shape[0], 5, 2, 1]

parameters = model(X, y, layers_dims, num_iterations=10000, print_cost=True)