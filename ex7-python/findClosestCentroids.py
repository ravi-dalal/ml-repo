import numpy as np

def findClosestCentroids(X, centroids):
	K = centroids.shape[0]
	idx = np.zeros((X.shape[0], 1))
	cost = np.zeros((K, 1))
	for i in range(X.shape[0]):
		for k in range(K):
			cost[k] = np.sum(np.square(X[i] - centroids[k]))
			#print(cost[k])
		idx[i] = np.argmin(cost)+1

	#print('idx = ', idx)
	return idx