# Import  the required libraries    
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
import scipy
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


