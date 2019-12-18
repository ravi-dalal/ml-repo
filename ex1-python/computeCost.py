import numpy as np

def computeCost(x, y, theta):
    m = len(y)
    h = np.dot(x, theta)
    J = np.sum(np.square(np.subtract(h,y))) / (2.0 * m)
    return J
