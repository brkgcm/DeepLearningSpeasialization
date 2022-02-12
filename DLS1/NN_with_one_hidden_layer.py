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

