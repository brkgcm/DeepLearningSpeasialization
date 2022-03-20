# Import  the required libraries    
import numpy as np
import copy


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initilize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


# Forward and backward propagation
def propagate(w, b, X, Y):
    # Forward Propagation
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A))) / (-m)

    # Backward Propagation
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(np.array(cost))

    grads = {'dw': dw, 'db': db}

    return grads, cost


# Optimiization
def optimize(w, b, X, Y, num_iteration=100, learning_rate=0.0009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(num_iteration):
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads['dw']
        db = grads['db']
        # Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        if print_cost:
            print('Cost after iteration %i: %f' % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iteration=2000, learning_rate=0.5, print_cost=False):
    w, b = initilize_with_zeros((X_train.shape[0], 1)), 0
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iteration, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iteration}

    return d
