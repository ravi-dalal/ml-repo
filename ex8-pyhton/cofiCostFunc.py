import numpy as np

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda, return_grad=False):

    # Unfold the U and W matrices from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features, order='F')
    Theta = params[num_movies*num_features:].reshape(num_users, num_features, order='F')
    
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    J = 0.5 * np.sum(np.square(np.dot(X, Theta.T) - Y) * R)
    J = J + np.sum((_lambda / 2) * np.square(Theta)) + np.sum((_lambda / 2) * np.square(X))
    
    X_grad = np.dot(((np.dot(X, Theta.T) - Y) * R), Theta)
    X_grad =  X_grad + (_lambda * X)
    
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X)
    Theta_grad = Theta_grad + (_lambda * Theta)
    
    print(J)   
    grad = np.concatenate((X_grad.reshape(X_grad.size, order='F'), Theta_grad.reshape(Theta_grad.size, order='F')))
    if return_grad:
        return J, grad
    else:
        return J
