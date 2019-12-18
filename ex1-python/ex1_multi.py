# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:55:50 2017

@author: Ravi
"""

from featureNormalize import featureNormalize
from gradientDescent import gradientDescent
from plotData import plotData
from computeCost import computeCost
from normalEqn import normalEqn

import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# ================ Part 1: Feature Normalization ================

print('Loading data ...\n')
data = np.loadtxt('data/ex1data2.txt',delimiter=',')
X = data[:,0:2]
m, n = X.shape
y = data[:,2].reshape(m,1)

X, mu, sigma = featureNormalize(X)
X = np.column_stack((np.ones((m,1)), X))
#print(mu)
#print(sigma)

# ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((n+1, 1))
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(0,J_history.shape[0]), J_history, '-b', 'LineWidth', 2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print('\n', theta)
print('\n')

area_norm = (1650 - float(mu[0])) / float(sigma[0])
br_norm = (3 - float(mu[1]))/float(sigma[1])
x1 = np.array([1, area_norm, br_norm])
#print(x1)
price = np.dot(x1, theta)
print('Predicted price of a 1650 sq-ft, 3 br house using gradient descent:' , price)

theta = normalEqn(X, y)
print('Theta computed from normal equation: \n')
print('\n', theta)
print('\n')

price = np.dot(x1, theta)
print('Predicted price of a 1650 sq-ft, 3 br house using normal equation:' , price)

