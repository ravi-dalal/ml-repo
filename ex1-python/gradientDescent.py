import numpy as np
from computeCost import computeCost

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(1, num_iters):
        h = np.dot(x, theta)
        delta = (alpha * (np.dot(np.subtract(h,y).T, x))) / m
        theta = np.subtract(theta, delta.T)
        J_history[i] = computeCost(x, y, theta)
    return theta, J_history
