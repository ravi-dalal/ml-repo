# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:15:32 2017

@author: Ravi Dalal
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import fmin_bfgs

def plotData(X, y):
    y1 = np.where(y == 1)
    y2 = np.where(y == 0)
    plt.scatter(X[y1,0], X[y1,1], marker='+', label='y=1')
    plt.scatter(X[y2,0], X[y2,1], marker='.', label='y=0')
    plt.legend(loc='upper right')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

def plotDecisionBoundary(theta, X, y):
    plotData(X, y)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(1,len(u)):
        for j in range(1,len(v)):
            z[i,j] = np.dot(mapFeature(np.asarray([[u[i]]]), np.asarray([[v[j]]])),theta)
    z = z.T
    print(z)
    plt.contour(u, v, z, levels=[0])
    
def mapFeature(X1, X2):
    r = X1.shape[0]
    degree = 6
    out = np.ones((r,1))
    for i in range(1,degree+1):
        for j in range (0,i+1):
            newcol = (np.multiply(np.power(X1,(i-j)),np.power(X2,j)))
            out = np.hstack((out, newcol))
    return out
    
def costFunction(theta, X, y, _lambda, return_grad=False):
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
    
def predict (theta, X):
    r,c = X.shape
    theta = theta.reshape(c,1)
    p = expit(np.dot(X, theta))
    return (p >= 0.5)
    
np.set_printoptions(suppress=True)
data = np.loadtxt('D:/Learn/ML/machine-learning-ex2/ex2/ex2data2.txt',delimiter=',')
x = data[:,:2]
X = x
y = data[:,2]
m,n = X.shape
y = y.reshape(m,1)
plotData (X,y)
plt.show()
X = mapFeature(X[:,0].reshape(m,1), X[:,1].reshape(m,1))
m,n = X.shape

initial_theta = np.zeros((n, 1))
_lambda = 1
cost, grad = costFunction(initial_theta, X, y, _lambda, return_grad=True)
print("Cost={}, Gradient={}".format(cost,grad[:5]))

test_theta = np.ones((n, 1))
cost, grad = costFunction(test_theta, X, y, 10, return_grad=True)
print("Cost={}, Gradient={}".format(cost,grad[:5]))

fargs=(X, y, 1)
theta = fmin_bfgs(costFunction, x0=initial_theta, args=fargs)
#print(theta)
plotDecisionBoundary(theta, x, y)
plt.show()
print("Train Accuracy: ",np.mean(predict(theta, X) == y)*100)