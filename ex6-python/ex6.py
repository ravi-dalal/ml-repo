import sys
import numpy as np
from scipy.io import loadmat
from plotData import plotData
from visualizeBoundaryLinear import visualizeBoundaryLinear
from svmTrain import svmTrain
from gaussianKernel import gaussianKernel
from visualizeBoundary import visualizeBoundary
from dataset3Params import dataset3Params

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# =========== Part 1: Loading and Visualizing Data =============
"""
data1 = loadmat("data\ex6data1.mat")
X = data1['X']
y = data1['y']

plt = plotData(X, y)
plt.show()

# ==================== Part 2: Training Linear SVM ====================
C = 1
model = svmTrain(X, y, C, 'linear')
visualizeBoundaryLinear(X, y, model)

# =============== Part 3: Implementing Gaussian Kernel ===============
x1 = [1, 2, 1]
x2 = [0, 4, -1]
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = \n(for sigma = 2, this value should be about 0.324652)\n', sim)

# =============== Part 4: Visualizing Dataset 2 ================
data2 = loadmat("data\ex6data2.mat")
X = data2['X']
y = data2['y']

#plt = plotData(X, y)
#plt.show()

C = 1
sigma = 0.1
# in sklearn SVC, gamma = 1/2*sigma^2
model = svmTrain(X, y, C, 'rbf', 1 / (2 * np.square(sigma)))
visualizeBoundary(X, y, model)
"""
# =============== Part 6: Visualizing Dataset 3 ================

data3 = loadmat("data\ex6data3.mat")
X = data3['X']
y = data3['y']
Xval = data3['Xval']
yval = data3['yval']

#plt = plotData(X, y)
#plt.show()

C, sigma = dataset3Params(X, y, Xval, yval)
#print('C=', C, 'Sigma=', sigma)
model = svmTrain(X, y, C, 'rbf', 1 / (2 * np.square(sigma)))
visualizeBoundary(X, y, model)
