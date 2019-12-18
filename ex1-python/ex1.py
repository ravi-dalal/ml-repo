# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:55:50 2017

@author: Ravi
"""

from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

   
np.set_printoptions(suppress=True)
# ==================== Part 1: Basic Function ====================
#print('Running warmUpExercise ... \n')
#print('5x5 Identity Matrix: \n');
#print(warmUpExercise(5))
#print('\n')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
X,y = np.loadtxt('data/ex1data1.txt',delimiter=',').T
plt = plotData(X, y)

# =================== Part 3: Cost and Gradient descent ===================
theta = np.zeros((2, 1))
x=np.vstack(X)
#print(x)
y=np.vstack(y)
x=np.hstack((np.ones((len(x),1)), x))
print('\nTesting the cost function ...\n')
J = computeCost(x, y, theta);
print('With theta = [0 ; 0]\nCost computed = \n', J);
print('Expected cost value (approx) 32.07\n');

# further testing of the cost function
J = computeCost(x, y, [[-1] , [2]])
print('\nWith theta = [-1 ; 2]\nCost computed = \n', J);
print('Expected cost value (approx) 54.24\n');

iterations = 1500
alpha = 0.01
#theta, costs = gradientDescent(x, y, theta, alpha, iterations)
# print theta to screen
print('Theta found by gradient descent:\n');
print('\n', theta);
print('Expected theta values (approx)\n');
print(' -3.6303\n  1.1664\n\n');

plt.plot(X,x.dot(theta),'r-', label='Linear regression')
plt.legend(loc='lower right')
plt.show()

predict1 = np.dot(np.array([1, 3.5]).reshape(1,2), theta)
print('For population = 35,000, we predict a profit of \n', predict1*10000);
predict2 = np.dot(np.array([1, 7]).reshape(1,2), theta)
print('For population = 70,000, we predict a profit of \n', predict2*10000);

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')
theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 100)
J = np.zeros((len(theta0), len(theta1)))
for i in range(1,len(theta0)):
    for j in range(1,len(theta1)):
        t = [[theta0[i]], [theta1[j]]]
        J[i,j] = computeCost(x, y, t)
J=J.T
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
theta0, theta1 = np.meshgrid(theta0, theta1)
ax.invert_xaxis()
surf = ax.plot_surface(theta0, theta1, J, cmap=mpl.cm.coolwarm, rstride=2, cstride=2)
fig.colorbar(surf)
plt.xlabel('Theta_0')
plt.ylabel('Theta_1')
plt.show()
fig = plt.figure()

cgraph = plt.contour(theta0, theta1, J, np.logspace(-2,3,20), cmap=mpl.cm.coolwarm)
fig.colorbar(cgraph)
plt.xlabel('Theta_0')
plt.ylabel('Theta_1')
plt.plot(theta[0,0], theta[1,0], 'rx')
plt.show()
