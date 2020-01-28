import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as img
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# ================= Part 1: Find Closest Centroids ====================
'''
data = loadmat("data/ex7data2.mat")
X = data['X']

K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)
print('Closest centroids for the first 3 examples: \n')
print(idx[0:3])
print('\n(the closest centroids should be 1, 3, 2 respectively)\n')

# ===================== Part 2: Compute Means =========================

print('\nComputing centroids means.\n\n');

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(centroids)
print('\n(the centroids should be\n');
print('   [ 2.428301 3.157924 ]\n');
print('   [ 5.813503 2.633656 ]\n');
print('   [ 7.119387 3.616684 ]\n\n');

# =================== Part 3: K-Means Clustering ======================

K = 3
max_iters = 10

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
'''
# ============= Part 4: K-Means Clustering on Pixels ===============

print('\nRunning K-Means clustering on pixels from an image.\n\n')

#  Load an image of a bird
A = np.double(img.imread('data/bird_small.png'))
#print('A = ', A)
# Divide by 255 so that all values are in the range 0 - 1, not needed with imread
#A = A / 255
img_size = A.shape
#print(img_size)

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

# ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

print('\nApplying K-Means to compress an image.\n\n')

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
X_recovered = centroids[idx.astype(int)-1,:]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image 
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title('Compressed, with '+str(K)+' colors.')
plt.show()
