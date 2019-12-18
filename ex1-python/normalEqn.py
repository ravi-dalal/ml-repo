import numpy as np
from numpy.linalg import pinv

def normalEqn(X, y):
    return np.dot(np.dot(pinv(np.dot(X.T, X)), X.T), y)