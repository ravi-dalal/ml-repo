import numpy as np
from svmTrain import svmTrain

def dataset3Params(X, y, Xval, yval):
	C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	scores = np.zeros(((len(C_values) * len(sigma_values)), 3))
	counter = 0
	for C in C_values:
		for sigma in sigma_values:
			gamma = 1 / (2 * np.square(sigma))
			model = svmTrain(X, y, C, 'rbf', gamma)			
			scores[counter, 0] = model.score(Xval, yval)
			scores[counter, 1] = C
			scores[counter, 2] = sigma
			counter = counter + 1

	max_score = np.argmax(scores, axis=0)
	return scores[max_score[0], 1], scores[max_score[0], 2]

