import numpy as np
from sigmoidFunc import sigmoidFunc

def costFunction(theta, X, y, return_grad=False):
    m = len(y)
    #The fmin function flattens the x0 argument, so needs to be reshaped in matrix
    r,c = X.shape
    theta = theta.reshape(c,1)
    #print("Theta=", theta)
    t1 = (-1.0 * y) * (np.log(sigmoidFunc(np.dot(X, theta))))
    t2 = (1.0 - y) * (np.log(1.0 - sigmoidFunc(np.dot(X, theta))))
    cost = ((1.0 / m) * (t1 - t2)).sum()
    #print (cost)
    gradient = (1.0 / m) * (np.dot((sigmoidFunc(np.dot(X, theta)) - y).T,X))
    #print (gradient)
    if return_grad:
        return cost, gradient
    else:
        return cost
