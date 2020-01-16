# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:32:46 2017

@author: Ravi Dalal
"""

from displayData import displayData
from predict import predict
import sys
import numpy as np
from scipy.io import loadmat

    
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
# =========== Part 1: Loading and Visualizing Data =============

# Load Training Data
print('Loading and Visualizing Data ...\n')
data = loadmat("data/ex3Data1.mat")
X = data['X']
y = data['y']
#print(y)
m, n = X.shape
rand_indices = np.random.permutation(m)
sel = np.take(X, rand_indices[0:100], 0)
plt, h, display_array = displayData(sel)
#plt.show()

# ================ Part 2: Loading Pameters ================
weights = loadmat("data/ex3weights.mat")
theta1 = weights['Theta1']
theta2 = weights['Theta2']
#print(X.shape)
#print(theta1.shape)
#print(theta2.shape)

# ================= Part 3: Implement Predict =================
pred = predict(theta1, theta2, X, y)
#print(pred.shape)
print("Train Accuracy: ",np.mean(pred == y)*100)

#  Randomly permute examples
rand_indices = np.random.permutation(X.shape[0])
r = rand_indices[0:m]
#print(r)
for i in range(0, m):
    #sel = np.take(X, r[i], 0)
    #print(X.shape)
    #print('r[i]]=', r[i])
    sel = X[r[i], :].reshape(1, n)
    #print("Selected=", sel.shape)
    #print(np.take(y,r, 0))
    plt, h, display_array = displayData(sel)
    plt.show()
    pred = predict(theta1, theta2, sel, y)
    print("\nNeural Network Prediction: \n", pred)
    print("digit \n", np.mod(pred, 10))
    # Pause with quit option
    s = input("Paused - press enter to continue, q to exit: ")
    if s == 'q':
        break
