# import numpy
import numpy as np

#Pad images with zeros
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad

# Single step of convolution
def cov_single_step(a_slice_prev, W, b)
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s, dtype=np.float64)
    Z = Z + np.asscalar(b)
    return Z