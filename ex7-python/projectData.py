import numpy as np

def projectData(X, U, K):
	Z = np.dot(X, U[:, 0:K])
	return Z