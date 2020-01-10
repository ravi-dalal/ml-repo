import numpy as np

def gaussianKernel (x1, x2, sigma):
	return np.exp(-1 * (np.sum(np.square(np.subtract(x1, x2))) / (2 * np.square(sigma))))