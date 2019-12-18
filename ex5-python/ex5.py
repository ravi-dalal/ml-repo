import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def linearRegCostFunction (theta, x, y, _lambda, return_grad=False):
    m, n = x.shape
    theta = theta.reshape(x.shape[1],1)
    h = np.dot(x, theta)
    J = np.sum(np.square(np.subtract(h,y))) / (2.0 * m)
    reg = (_lambda * np.sum(np.square(theta[1:]))) / (2.0 * m)
    J = J + reg
    grad = np.zeros(theta.shape)
    x0 = x[:,[0]]
    x1 = x[:,1:]
    grad[1:,:] = (np.sum(np.dot(np.subtract(h,y).T, x1)) / m ) + (_lambda * theta[1:,:]) / m
    grad[0,:] = (np.sum(np.dot(np.subtract(h,y).T, x0)) / m )
    if return_grad:
        return J, grad
    else:
        return J

def trainLinearReg (x, y , _lambda):
    m, n = x.shape
    fargs = (x, y, _lambda)
    initial_theta = np.zeros((x.shape[1], 1))
    return minimize(linearRegCostFunction, x0=initial_theta, args=fargs, method="CG", options={'disp': False, 'maxiter': 50.0}, tol=0.00000000005)

def learningCurve (x, y, xval, yval, _lambda):
    m, n = x.shape
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    for i in range(0, m):
        x1 = x[0:i+1,:]
        y1 = y[0:i+1,:]
        result = trainLinearReg (x1, y1, _lambda)
        theta = result.x
        error_train[i] = linearRegCostFunction(theta, x1, y1, _lambda)
        error_val[i] = linearRegCostFunction(theta, xval, yval, _lambda)
    return error_train, error_val
    
def polyFeatures(x, p):
    x_poly = x
    for i in range (1, p):
        x_poly = np.column_stack((x_poly, np.power(x,i+1)))
    return x_poly

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm/sigma
    return X_norm, mu, sigma

def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05))
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)
    
#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

data = loadmat("..\machine-learning-ex5\ex5\ex5data1.mat")
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m,n = X.shape

#theta = np.ones((2,1))
#print (linearRegCostFunction(np.column_stack((np.ones((m,1)), X)), y, theta, 1))
#print (linearRegCostFunction(np.column_stack((np.ones((m,1)), X)), y, theta, 1, True))
#result = trainLinearReg(np.column_stack((np.ones((m,1)), X)), y, 0)
#plt.scatter(X, y, marker='x')
#plt.ylabel('Water flowing out of the dam (y)')
#plt.xlabel('Change in water level (x)')
#plt.xticks(np.arange(-50, 50, 10.0))
#plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), result.x))
#plt.show()

#error_train, error_val = learningCurve(np.column_stack((np.ones((m,1)), X)), y, np.column_stack((np.ones((Xval.shape[0],1)), Xval)), yval, 0)
#plt.plot(range(0, m), error_train, label="Training Error")
#plt.plot(range(0, m), error_val, label="Validation Error")
#plt.legend()
#plt.xlabel('Number of training examples')
#plt.ylabel('Error')
#plt.show()
p = 8
x_poly = polyFeatures(X, p)
x_poly, mu, sigma = featureNormalize(x_poly)
x_poly = np.column_stack((np.ones((x_poly.shape[0],1)), x_poly))

x_poly_test = polyFeatures(Xtest, p)
x_poly_test = x_poly_test - mu
x_poly_test = x_poly_test / sigma
x_poly_test = np.column_stack((np.ones((x_poly_test.shape[0],1)), x_poly_test))

x_poly_val = polyFeatures(Xval, p)
x_poly_val = x_poly_val - mu
x_poly_val = x_poly_val / sigma
x_poly_val = np.column_stack((np.ones((x_poly_val.shape[0],1)), x_poly_val))
_lambda = 0
result = trainLinearReg(x_poly, y, _lambda)
theta = result.x
plt.close()
plt.figure(1)
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)') 
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit (lambda = {:f})'.format(_lambda))
#plt.show()

plt.figure(2)
error_train, error_val = learningCurve(x_poly, y, x_poly_val, yval, 0)
p1, p2 = plt.plot(range(1,m+1), error_train, range(1,m+1), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(_lambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 50])
plt.legend((p1, p2), ('Train', 'Cross Validation'))
#plt.show()
#print('Polynomial Regression (lambda = {:f})\n\n'.format(_lambda))
#print('# Training Examples\tTrain Error\tCross Validation Error\n')
#for i in range(0, m):
#    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))
