import numpy as np

def computeCentroids(X, idx, K):
	m, n = X.shape
	centroids = np.zeros((K, n))
	for k in range(K):
		idx1, idx2 = np.nonzero(idx==(k+1))
		centroids[k,:] = np.mean(X[idx1,:], axis=0)
	return centroids
