# Import  the required libraries    
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import copy
from PIL import Image
from scipy import ndimage

def sigmoid(z):
    return 1/(1+np.exp(-z))


def initilize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

# Forward and backward propagation
def propagate(w, b, X, Y):
    #Forward Propagation
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = np.sum((Y*np.log(A))+((1-Y)*np.log(1-A)))/(-m)

    #Backward Propagation
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A-Y)/m

    cost = np.squeeze(np.array(cost))

    grads = {'dw':dw, 'db':db}

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
        w = w - learning_rate*dw
        b = b - learning_rate*db
        # Record the costs
        if i % 100 == 0:
            cost.append(cost)
        if print_cost:
            print('Cost after iteration %i: %f' %(i, cost))
        

    params = {'w':w, 'b':b}
    grads = {'dw':dw, 'db':db}

    return params, grads, costs


