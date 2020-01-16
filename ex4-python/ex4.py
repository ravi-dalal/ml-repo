# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:32:46 2017

@author: Ravi Dalal
"""

from randInitializeWeights import randInitializeWeights
from nnCostFunction import nnCostFunction
from vectorToArray import vectorToArray
from arrayToVector import arrayToVector
from checkNNGradients import checkNNGradients
from predict import predict
import sys
import numpy as np
from scipy.io import loadmat
#from scipy.optimize import fmin_cg
from scipy.optimize import minimize


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
data = loadmat("data/ex4data1.mat")
X = data['X']
y = data['y']
weights = loadmat("data/ex4weights.mat")
theta1 = weights['Theta1']
theta2 = weights['Theta2']
nn_params = arrayToVector(theta1, theta2)

# Feedforward using NN
print('\nFeedforward Using Neural Network ...\n')
_lambda = 0
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
print('Cost at parameters (loaded from ex4weights): \n(this value should be about 0.287629)\n', J)

_lambda = 1
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
print('Cost at parameters (loaded from ex4weights): \n(this value should be about 0.383770)\n', J)
"""
input_layer_size  = 9
hidden_layer_size = 3
num_labels = 4
m = 3
X  = debugInitializeWeights(m, input_layer_size - 1)
y = 1 + np.mod(range(m), num_labels).T
_lambda = 1
"""
_lambda = 3
#checkNNGradients(_lambda)
# Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
print('\nCost at (fixed) debugging parameters (for lambda = 3, this value should be about 0.576051)\n', debug_J)


# Training NN
initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_params = arrayToVector(initial_theta1, initial_theta2)
#nnCostFunction(initial_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

_lambda = 1
fargs = (input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
#params = fmin_cg(nnCostFunction, x0=initial_params, args=fargs, maxiter=10, gtol=1e-3)
result = minimize(nnCostFunction, x0=initial_params, args=fargs, method="CG", options={'disp': True, 'maxiter': 1.0}, tol=0.00000000005)
params = result.x
theta1, theta2 = vectorToArray (params, hidden_layer_size, input_layer_size, num_labels)

# Prediction based on trained NN
prediction = predict(theta1, theta2, X, y)
accuracy = np.mean(np.double(prediction == y)) * 100
print('Training Set Accuracy: %f' %accuracy)
