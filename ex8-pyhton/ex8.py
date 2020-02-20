import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit

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
plt.show()