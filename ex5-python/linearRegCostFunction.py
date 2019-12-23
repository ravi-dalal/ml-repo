import numpy as np

def linearRegCostFunction (theta, x, y, _lambda, return_grad=False):
    m, n = x.shape
    theta = theta.reshape(n, 1)
    h = np.dot(x, theta)
    J = np.sum(np.square(np.subtract(h,y))) / (2.0 * m)
    reg = (_lambda * np.sum(np.square(theta[1:]))) / (2.0 * m)
    J = J + reg
    grad = np.zeros(theta.shape)
    x0 = x[:,[0]]
    x1 = x[:,1:]
    grad[1:,:] = (np.sum(np.dot(np.subtract(h,y).T, x1)) / m ) + (_lambda * theta[1:,:]) / m
    grad[0,:] = (np.sum(np.dot(np.subtract(h,y).T, x0)) / m )
    if return_grad:
        return J, grad
    else:
        return J
