import numpy as np
from computeNumericalGradient import computeNumericalGradient
from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnCostFunction
from arrayToVector import arrayToVector
from decimal import Decimal


def checkNNGradients(_lambda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(range(m), num_labels).T
    nn_params = arrayToVector(theta1, theta2)
    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, _lambda, True)

    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    print('Numerical Gradient', 'Analytical Gradient')
    for numerical, analytical in zip(numgrad, grad):
        print(numerical, analytical)

    print('The above two columns you get should be very similar.\n' \
             '(Left Col.: Your Numerical Gradient, Right Col.: Analytical Gradient)')

    diff = Decimal(np.linalg.norm(numgrad-grad))/Decimal(np.linalg.norm(numgrad+grad))
    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). \n' \
             '\nRelative Difference: {:.10E}'.format(diff))
