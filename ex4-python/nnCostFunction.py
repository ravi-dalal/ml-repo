from arrayToVector import arrayToVector
from vectorToArray import vectorToArray
from sigmoidGradient import sigmoidGradient
import numpy as np
from scipy.special import expit

#counter = 0

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
    print('Cost from cost Function: {:.6f}'.format(cost))

#    global counter
#    counter += 1 
#    print("Running=", counter)

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
