import numpy as np
from sigmoidFunc import sigmoidFunc

def predict (theta, X):
    r,c = X.shape
    theta = theta.reshape(c,1)
    p = sigmoidFunc(np.dot(X, theta))
    return (p >= 0.5)
