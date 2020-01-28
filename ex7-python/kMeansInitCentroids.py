import numpy as np

def kMeansInitCentroids(X, K):
	centroids = np.zeros((K, X.shape[1]))
	randidx = np.random.permutation(X.shape[0])
	centroids = X[randidx[:K], :]
	return centroids