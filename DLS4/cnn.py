# import numpy
import numpy as np

#Pad images with zeros
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (1, 1), (3, 3), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad
