import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as img
from featureNormalize import featureNormalize
from pca import pca
from drawLine import drawLine
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
from hsv import hsv
import matplotlib.image as img
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from plotDataPoints import plotDataPoints

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
#np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# ================== Part 1: Load Example Dataset  ===================
'''
data = loadmat("data/ex7data1.mat")
X = data['X']
#print('X = ', X.shape)
#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo', linestyle='None', mfc='none')
#plt.show()

# =============== Part 2: Principal Component Analysis ===============

X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U, S = pca(X_norm)

drawLine(mu, mu + 1.5 * S[0] * U[:,0].T, color='k')
drawLine(mu, mu + 1.5 * S[1] * U[:,1].T, color='k')
plt.show()
plt.close()

print('Top eigenvector: \n')
print('{: 0.6f}'.format(U[0,0]), '{: 0.6f}'.format(U[1,0]))
print('\n(you should expect to see -0.707107 -0.707107)\n')

# =================== Part 3: Dimension Reduction ===================

plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo', linestyle='None', mfc='none')
#plt.show()

K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: ', '{: 0.6f}'.format(Z[0,0]))
print('(this value should be about 1.481274)\n')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: ', '{: 0.6f}'.format(X_rec[0, 0]), '{: 0.6f}'.format(X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro', linestyle='None', mfc='none')
for i in range(X_norm.shape[0]):
	drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)
plt.show()

# =============== Part 4: Loading and Visualizing Face Data =============

data = loadmat("data/ex7faces.mat")
X = data['X']
#  Display the first 100 faces in the dataset
#displayData(X[0:100, :])
#plt.show()

# =========== Part 5: PCA on Face Data: Eigenfaces  ===================

X_norm, mu, sigma = featureNormalize(X)
#  Run PCA
U, S = pca(X_norm)
#  Visualize the top 36 eigenvectors found
#displayData(U[:, 0:36].T)
#plt.show()

# ============= Part 6: Dimension Reduction for Faces =================

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(Z.shape)

# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====

#K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
plt.subplot(1, 2, 1)
displayData(X_norm[0:100,:])
plt.title('Original faces')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayData(X_rec[0:100,:])
plt.title('Recovered faces')
plt.show()
'''
# === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===

A = np.double(img.imread('data/bird_small.png'))
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

sel = (np.floor(np.random.rand(1000, 1) * X.shape[0]) + 1).astype(int).flatten()

#  Setup Color Palette
palette = hsv(K + 1)
colors = palette[idx[sel].flatten().astype(int), :]

#  Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=10, edgecolors=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()

# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
[U, S] = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.close()
plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
