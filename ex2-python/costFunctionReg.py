import numpy as np
from scipy.special import expit

def costFunctionReg(theta, X, y, _lambda, return_grad=False):
    m = len(y)
    r,c = X.shape
    theta = theta.reshape(c,1)
    t1 = (-1.0 * y) * (np.log(expit(np.dot(X, theta))))
    t2 = (1.0 - y) * (np.log(1.0 - expit(np.dot(X, theta))))
    t3 = (1.0 / m) * ((t1 - t2).sum())
    t4 = (_lambda/(2*m)) * ((np.dot(theta.T,theta) - theta[0]**2).sum())
    cost = t3 + t4
    grad = ((1.0 / m) * (np.dot((expit(np.dot(X, theta)) - y).T,X))).reshape(c,1)
    #print(grad)
    reg = (_lambda/m) * theta
    #print(reg)
    gradient = grad + reg
    gradient[0,0] = grad[0,0]
    #print (gradient)
    if return_grad:
        return cost, gradient
    else:
        return cost 
