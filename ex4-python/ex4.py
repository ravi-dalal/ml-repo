import sys
import numpy as np
from scipy.io import loadmat
from scipy.special import expit
#from scipy.optimize import fmin_cg
from scipy.optimize import minimize
from decimal import Decimal

counter = 0

def arrayToVector (a, b):
    return np.concatenate((a, b),axis=None)
    #return np.concatenate((a.reshape(a.size, order='F'), b.reshape(b.size, order='F')))
    #return np.concatenate((np.ravel(a, order='F'), np.ravel(b, order='F')))
    
def vectorToArray (vector, hidden_layer_size, input_layer_size, num_labels):
    theta1 = vector[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
    theta2 = vector[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size +1))
    #theta1 = np.reshape(vector[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
    #theta2 = np.reshape(vector[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')
    return theta1, theta2
    
def sigmoidGradient (z):
    sigGrad = np.multiply (expit(z), (1.0 - expit(z)))
    return sigGrad

def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W
    
def computeNumericalGradient(J, theta):
    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0
    return numgrad

def debugInitializeWeights(fan_out, fan_in):
    # Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10
    return W    
    
def checkNNGradients(lambda_reg=0):
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
                   num_labels, X, y, lambda_reg, True)

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
             
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, 
                                   X, y, _lambda, return_grad=False):
    theta1, theta2 = vectorToArray (nn_params, hidden_layer_size, input_layer_size, num_labels)
    m,n = X.shape
    a1 = np.column_stack((np.ones((m,1)), X))   
    a2 = expit(np.dot(a1, theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
    h = expit(np.dot(a2, theta2.T))
    cost = 0;
    Y = np.identity(num_labels)[y-1,:].reshape(m,num_labels)
    t1 = -1.0 * Y
    t1 *= np.log(h)
    t2 = (1.0 - Y) 
    t2 *= (np.log(1.0 - h))
    cost = (1.0 / m) * ((t1 - t2).sum())
    #print('{:.6f}'.format(cost))
    #print('theta1=', theta1.shape)
    #print('theta2=', theta2.shape)
    theta1NoBias = theta1[:,1:]
    theta2NoBias = theta2[:,1:]
    #print('theta1NoBias=', theta1NoBias.shape)
    #print('theta2NoBias=', theta2NoBias.shape)
    reg = _lambda/(2.0*m)
    reg *= np.square(theta1NoBias).sum() + np.square(theta2NoBias).sum()
    cost += reg
    #print('{:.6f}'.format(cost))
    global counter
    counter += 1 
    print("Running=", counter)
    #Backpropagation using vectorization
    d3 = h - Y
    d2 = np.multiply(np.dot(d3, theta2NoBias) , sigmoidGradient(np.dot(a1, theta1.T)))
    delta1 = np.dot(d2.T, a1)
    delta2 = np.dot(d3.T, a2)
    #print("delta1=", delta1)
    #print("delta2=", delta2)
    theta1_grad = (1.0 / m) * delta1
    theta2_grad = (1.0 / m) * delta2
    #print(theta1_grad.shape)
    #print(theta2_grad.shape)
    #print('delta1=', delta1.shape)
    #print('delta2=', delta2.shape)
    """
    dt1 = 0
    dt2 = 0
    for t in range (0, m):
        #Step 1
        a1 = np.column_stack((np.ones((1,1)), X[t,:].reshape(1,n)))
        #print('a1=', a1.shape)
        a2 = expit(np.dot(a1, theta1.T))
        #print('a2=', a2.shape)
        a2 = np.column_stack((np.ones((1,1)), a2))
        a3 = expit(np.dot(a2, theta2.T))
        #print('a3=', a3.shape)
        #Step 2
        d3 = a3 - Y[t, :].T
        #print('d3=',d3.shape)
        #Step 3
        d2 = np.multiply(np.dot(d3, theta2NoBias) , sigmoidGradient(np.dot(a1, theta1.T)))
        #print('d2=', d2.shape)
        #Step 4
        dt2 += np.dot(d3.T, a2)
        dt1 += np.dot(d2.T, a1)
    print("dt1=", dt1)
    print("dt2=", dt2)
    """  
    # Regularization
    # only regularize for j >= 1, so skip the first column
    theta1_grad_unregularized = np.copy(theta1_grad)
    theta2_grad_unregularized = np.copy(theta2_grad)
    theta1_grad += (float(_lambda)/m)*theta1
    theta2_grad += (float(_lambda)/m)*theta2
    theta1_grad[:,0] = theta1_grad_unregularized[:,0]
    theta2_grad[:,0] = theta2_grad_unregularized[:,0]

    gradient = arrayToVector(theta1_grad, theta2_grad)
    if return_grad:
        return cost, gradient
    else:
        return cost
        
def predict (theta1, theta2, X, y):
    #print(X)
    m, n = X.shape
    a1 = np.column_stack((np.ones((m,1)), X))   
    a2 = expit(np.dot(a1, theta1.T))
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))
    #print(a2.shape)
    a3 = expit(np.dot(a2, theta2.T))
    #print(a3.shape)
    #p = np.max(a3, axis=1)
    #p = p.reshape(m,1)
    #print(a3)
    pindex = np.argmax(a3, axis=1)+1
    pindex = pindex.reshape(m,1)
    #print(pindex)
    return pindex        

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
data = loadmat("..\machine-learning-ex4\ex4\ex4data1.mat")
X = data['X']
y = data['y']
weights = loadmat("..\machine-learning-ex4\ex4\ex4weights.mat")
theta1 = weights['Theta1']
theta2 = weights['Theta2']
_lambda = 1
nn_params = arrayToVector(theta1, theta2)
nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
checkNNGradients()
"""
input_layer_size  = 9
hidden_layer_size = 3
num_labels = 4
m = 3
X  = debugInitializeWeights(m, input_layer_size - 1)
y = 1 + np.mod(range(m), num_labels).T
_lambda = 1
"""
initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_params = arrayToVector(initial_theta1, initial_theta2)
nnCostFunction(initial_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

fargs = (input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
#params = fmin_cg(nnCostFunction, x0=initial_params, args=fargs, maxiter=10, gtol=1e-3)
result = minimize(nnCostFunction, x0=initial_params, args=fargs, method="CG", options={'disp': True, 'maxiter': 50.0}, tol=0.00000000005)
params = result.x
theta1_min, theta2_min = vectorToArray (params, hidden_layer_size, input_layer_size, num_labels)
prediction = predict(theta1_min, theta2_min, X, y)
accuracy = np.mean(np.double(prediction == y)) * 100
print('Training Set Accuracy: %f' %accuracy)
