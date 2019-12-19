# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:32:46 2017

@author: Ravi Dalal
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.io import loadmat
#from scipy.optimize import fmin_cg
from scipy.optimize import minimize

def displayData (X):
    example_width = int(round(math.sqrt(np.size(X,1))))
    plt.gray()
    m,n = X.shape
    example_height = int(n / example_width)
    
    display_rows = int(math.floor(np.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))
    pad = 1;
    
    display_array = np.ones((pad + display_rows * (example_height + pad), \
                             pad + display_cols * (example_width + pad)))
    curr_ex = 0
    for j in range(0, display_rows):
        for i in range (0, display_cols):
            if curr_ex >= m:
                break
            max_val = max(abs(X[curr_ex, :]))
            rows = pad + j * (example_height + pad) + np.array(range(example_height))
            cols = pad + i * (example_width + pad) + np.array(range(example_width))
            display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = \
            np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") \
            / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
            break
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    # Do not show axis
    plt.axis('off')
    plt.show(block=False)
    return h, display_array

def lrCostFunction(theta, X, y, _lambda, return_grad=False):
    m = len(y)
    r,c = X.shape
    theta = theta.reshape(c,1)
    t1 = (-1.0 * y) * (np.log(expit(np.dot(X, theta))))
    t2 = (1.0 - y) * (np.log(1.0 - expit(np.dot(X, theta))))
    t3 = (1.0 / m) * ((t1 - t2).sum())
    t4 = (_lambda/(2*m)) * ((np.dot(theta.T,theta) - theta[0]**2).sum())
    cost = t3 + t4
    grad = ((1.0 / m) * (np.dot((expit(np.dot(X, theta)) - y).T,X))).reshape(c,1)
    #print(cost)
    reg = (_lambda/m) * theta
    #print(reg)
    gradient = grad + reg
    gradient[0,0] = grad[0,0]
    #print (gradient)
    if return_grad:
        return cost, gradient
    else:
        return cost

def oneVsAll(X, y, num_labels, _lambda):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.column_stack((np.ones((m,1)), X))    
    for c in range(1, num_labels+1):
        #print(y==c)
        initial_theta = np.zeros((n + 1, 1)).flatten()
        print("Training {:d} out of {:d} categories...".format(c, num_labels))
        fargs = (X, (y==c).reshape(m,1), _lambda)
        theta = minimize(lrCostFunction, x0=initial_theta, args=fargs, method="CG")
                         #, options={'maxiter':10},\
                         #method="CG", tol=0.005)
        #print(theta)
        all_theta[c-1,:] = theta["x"]
    return all_theta
        
def predictOneVsAll(all_theta, X):
    m,n = X.shape
    k = all_theta.shape[0]
    theta = all_theta.reshape(k, n+1)
    X = np.column_stack((np.ones((m,1)), X))
    p = np.argmax(expit( np.dot(X,theta.T) ), axis=1)+1
    return p.reshape(m,1)
    
        
np.set_printoptions(suppress=True)
data = loadmat("data\ex3Data1.mat")
X = data['X']
y = data['y']
#print(X.shape)
#print(y.shape)
#print(np.unique(y))
m = np.size(X, 0)
rand_indices = np.random.permutation(m)
sel = np.take(X, rand_indices[0:100], 0)
displayData(sel)

#theta_t = np.array([-2, -1, 1, 2]).reshape(4,1)
#print(theta_t)
#X_t = np.column_stack((np.ones((5,1)), (np.reshape(np.arange(1,16),(5,3),order="F")/10)))
#print(X_t)
#y_t = np.array([1,0,1,0,1]).reshape(5,1)
#lambda_t = 3
#J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t, return_grad=True)
#print(J, grad)

_lambda = 0.1

all_theta = oneVsAll(X, y, 10, _lambda)
#print(all_theta.shape)
p = predictOneVsAll(all_theta, X)
#print(p.shape)
print("Train Accuracy: ",np.mean(p == y)*100)
