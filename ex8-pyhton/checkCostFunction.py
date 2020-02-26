import numpy as np
from cofiCostFunc import cofiCostFunc
from computeNumericalGradient import computeNumericalGradient
from decimal import Decimal


def checkCostFunction(_lambda=0):
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
    def costFunc(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, _lambda, True)

    _, grad = costFunc(params)
    numgrad = computeNumericalGradient(costFunc, params)

    print('Numerical Gradient', 'Analytical Gradient')
    for numerical, analytical in zip(numgrad, grad):
        print(numerical, analytical)

    print('The above two columns you get should be very similar.\n' \
             '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')

    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))
    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             '\nRelative Difference: {:.10E}'.format(diff))
