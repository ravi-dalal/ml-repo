# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:15:32 2017

@author: Ravi Dalal
"""
from costFunctionReg import costFunctionReg
from predict import predict
from mapFeature import mapFeature
from plotDecisionBoundary import plotDecisionBoundary
from plotData import plotData
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

data = np.loadtxt('data/ex2data2.txt',delimiter=',')
X = data[:,:2]
y = data[:,2:3]
m,n = X.shape
plt, pos, neg = plotData (X,y)
plt.legend((pos, neg), ('y=1', 'y=0'), loc='upper right')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

X = mapFeature(X[:,0].reshape(m,1), X[:,1].reshape(m,1))
print("Mapped X=", X.shape)
m,n = X.shape

initial_theta = np.zeros((n, 1))
_lambda = 1
cost, grad = costFunctionReg(initial_theta, X, y, _lambda, return_grad=True)
print("Cost={}, Gradient={}".format(cost,grad[:5]))

test_theta = np.ones((n, 1))
cost, grad = costFunctionReg(test_theta, X, y, 10, return_grad=True)
print("Cost={}, Gradient={}".format(cost,grad[:5]))

fargs=(X, y, _lambda)
theta = fmin_bfgs(costFunctionReg, x0=initial_theta, args=fargs)
#print(theta)
plotDecisionBoundary(theta, X, y)
plt.title("lambda = "+str(_lambda))
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
print("Train Accuracy: ",np.mean(predict(theta, X) == y)*100)

_lambda = 0
fargs=(X, y, _lambda)
theta = fmin_bfgs(costFunctionReg, x0=initial_theta, args=fargs)
plotDecisionBoundary(theta, X, y)
plt.title("lambda = "+str(_lambda))
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

_lambda = 100
fargs=(X, y, _lambda)
theta = fmin_bfgs(costFunctionReg, x0=initial_theta, args=fargs)
plotDecisionBoundary(theta, X, y)
plt.title("lambda = "+str(_lambda))
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()