import numpy as np

def recoverData(Z, U, K):
	X_rec = np.dot(Z, U[:, 0:K].T)
	return X_rec	