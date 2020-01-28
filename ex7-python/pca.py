import numpy as np
from numpy.linalg import svd

def pca(X):

	m, n = X.shape
	sigma = np.dot(X.T, X) / m
	U, S, V = svd(sigma)
	return U, S

