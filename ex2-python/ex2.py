# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:29:25 2017

@author: Ravi Dalal
"""

from plotData import plotData
from costFunction import costFunction
from sigmoidFunc import sigmoidFunc
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict
import numpy as np
from scipy.optimize import fmin
#from scipy.optimize import fmin_bfgs
  

# ==================== Part 1: Plotting ====================
data = np.loadtxt('data/ex2data1.txt',delimiter=',')
X = data[:,:2]
y = data[:,2:3]
plt, pos, neg = plotData (X,y)
plt.legend((pos, neg), ('Admitted', 'Not Admitted'), loc='lower left')
plt.ylabel('Exam 2 Score')
plt.xlabel('Exam 1 Score')
plt.show()

# ============ Part 2: Compute Cost and Gradient ============
m,n = X.shape
X = np.column_stack((np.ones((m,1)), X))
theta = np.zeros((n+1, 1))
cost, grad = costFunction(theta, X, y,return_grad=True)
print('Cost at initial theta (zeros): \n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' \n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test_theta = np.asarray([[-24], [0.2],[0.2]])
cost, grad = costFunction(test_theta, X, y,return_grad=True)
print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' \n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

# ============= Part 3: Optimizing using fmin  =============
fargs=(X, y)
theta = fmin(costFunction, x0=theta, args=fargs, maxiter=400)
#theta = fmin_bfgs(costFunction, x0=theta, args=fargs)
#print(theta)
print('Cost at theta found by fminunc: \n', cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' \n', theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

plotDecisionBoundary(theta, X, y)
plt.show()

# ============== Part 4: Predict and Accuracies ==============

prob = sigmoidFunc(np.dot(np.array([1, 45, 85]), theta))
print('For a student with scores 45 and 85, we predict an admission probability of \n', prob)
print('Expected value: 0.775 +/- 0.002\n\n')

print("Train Accuracy: ",np.mean(predict(theta, X) == y)*100)
print('Expected accuracy (approx): 89.0\n')
