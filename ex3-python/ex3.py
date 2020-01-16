# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:32:46 2017

@author: Ravi Dalal
"""

from lrCostFunction import lrCostFunction
from displayData import displayData
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll
import numpy as np
from scipy.io import loadmat
#from scipy.optimize import fmin_cg
    
        
np.set_printoptions(suppress=True)

# =========== Part 1: Loading and Visualizing Data =============
print('Loading and Visualizing Data ...\n')
data = loadmat("data/ex3Data1.mat")
X = data['X']
y = data['y']
#print(X.shape)
#print(y.shape)
#print(np.unique(y))
m = np.size(X, 0)
rand_indices = np.random.permutation(m)
sel = np.take(X, rand_indices[0:100], 0)
plt, h, display_array = displayData(sel)
plt.show()

# ============ Part 2a: Vectorize Logistic Regression ============
# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')
theta_t = np.array([-2, -1, 1, 2]).reshape(4,1)
#print(theta_t)
X_t = np.column_stack((np.ones((5,1)), (np.reshape(np.arange(1,16),(5,3),order="F")/10)))
#print(X_t)
y_t = np.array([1,0,1,0,1]).reshape(5,1)
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t, return_grad=True)
print('\nCost: \n', J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' \n', grad);
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

# ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

_lambda = 0.1
all_theta = oneVsAll(X, y, 10, _lambda)
#print(all_theta.shape)

# ================ Part 3: Predict for One-Vs-All ================
p = predictOneVsAll(all_theta, X)
#print(p.shape)
print("Train Accuracy: ",np.mean(p == y)*100)
