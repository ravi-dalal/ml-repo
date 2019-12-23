import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction

def validationCurve (x, y, xval, yval):
	lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
	error_train = np.zeros((len(lambda_vec), 1))
	error_val = np.zeros((len(lambda_vec), 1))
	theta = np.zeros((len(lambda_vec), x.shape[1]))
	
	for i in range(0, len(lambda_vec)):
		result = trainLinearReg(x, y, lambda_vec[i])
		#print(result.x)
		theta[i] = result.x
		result_val = trainLinearReg(xval, yval, lambda_vec[i])
		error_train[i] = linearRegCostFunction(theta[i], x, y, 0)
		error_val[i] = linearRegCostFunction(theta[i], xval, yval, 0)

	return lambda_vec, error_train, error_val, theta
