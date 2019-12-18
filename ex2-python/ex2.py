# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:29:25 2017

@author: Ravi Dalal
"""

import numpy as np
import matplotlib.pyplot as plt
#scipy.optimize.expit is implementaiton of sigmoid function
#from scipy.special import expit
from scipy.optimize import fmin
#from scipy.optimize import fmin_bfgs

def plotData(X, y):
    y1 = np.where(y == 1)
    y2 = np.where(y == 0)
    plt.scatter(X[y1,0], X[y1,1], marker='+', label='Admitted')
    plt.scatter(X[y2,0], X[y2,1], marker='.', label='Not Admitted')
    plt.legend(loc='lower left')
    plt.ylabel('Exam 2 Score')
    plt.xlabel('Exam 1 Score')
    #plt.show()
    
def plotDecisionBoundary (theta, X, y):
    plotData(X, y)
    r,c = X.shape
    if c <= 3:
        plot_x = np.array([min(X[:,1])-2,  max(X[:,1])+2])
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])
        plt.plot(plot_x, plot_y, label="Decision Boundary")
        plt.legend(loc='lower left')
        plt.axis([30, 100, 30, 100])

def sigmoidFunc(x):
    return np.divide(1.0, (1.0 + np.exp(-x)))

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
    
def predict (theta, X):
    r,c = X.shape
    theta = theta.reshape(c,1)
    p = sigmoidFunc(np.dot(X, theta))
    return (p >= 0.5)

#x1,x2,y = np.loadtxt('D:/Learn/ML/machine-learning-ex2/ex2/ex2data1.txt',delimiter=',').T
data = np.loadtxt('D:/Learn/ML/machine-learning-ex2/ex2/ex2data1.txt',delimiter=',')
x = data[:,:2]
X = x
y = data[:,2]
y = y.reshape(100,1)
plotData (X,y)
plt.show()
"""
splits = np.hsplit(data,[2, 3])
print(splits[1])
X = splits[0]
y = splits[1]
print(sigmoidFunc(y))
print("Expit=",expit(y))
"""
m,n = X.shape
X = np.column_stack((np.ones((m,1)), X))
theta = np.zeros((n+1, 1))
cost, grad = costFunction(theta, X, y,return_grad=True)
print("Cost={}, Gradient={}".format(cost,grad))
test_theta = np.asarray([[-24], [0.2],[0.2]])
cost, grad = costFunction(test_theta, X, y,return_grad=True)
print("Cost={}, Gradient={}".format(cost,grad))

fargs=(X, y)
theta = fmin(costFunction, x0=theta, args=fargs, maxiter=400)
#theta = fmin_bfgs(costFunction, x0=theta, args=fargs)
#print(theta)
plotDecisionBoundary(theta, x, y)
plt.show()
print("Train Accuracy: ",np.mean(predict(theta, X) == y)*100)