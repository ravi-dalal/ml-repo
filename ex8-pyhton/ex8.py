import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# ================== Part 1: Load Example Dataset  ===================

data = loadmat("data/ex8data1.mat")
X = data['X']
#plt.plot(X[:, 0], X[:, 1], 'bx')
#plt.axis([0, 30, 0, 30])
#plt.xlabel('Latency (ms)')
#plt.ylabel('Throughput (mb/s)')
#plt.show()

# ================== Part 2: Estimate the dataset statistics ===================

mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)

visualizeFit(X,  mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
#plt.show()

# ================== Part 3: Find Outliers ===================

Xval = data['Xval']
yval = data['yval']
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print('epsilon = ',epsilon)
print('F1 = ', F1)
outliers = (p < epsilon)
#print(outliers)
#  Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', LineWidth=2, MarkerSize=10, fillstyle='none')
plt.show()

# ================== Part 4: Multidimensional Outliers ===================

data2 = loadmat('data/ex8data2.mat')
X = data2['X']
Xval = data2['Xval']
yval = data2['yval']

mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)

print('Best epsilon found using cross-validation: ', epsilon)
print('Best F1 on Cross Validation Set:  ', F1)
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found: ', sum(p < epsilon))
