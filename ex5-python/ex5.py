import sys
import numpy as np
from scipy.io import loadmat
from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from validationCurve import validationCurve
from plotFit import plotFit
import matplotlib.pyplot as plt
    
#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# =========== Part 1: Loading and Visualizing Data =============
data = loadmat("data/ex5data1.mat")
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m,n = X.shape

plt.plot(X, y, marker='x', linestyle='None')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlabel('Change in water level (x)')
plt.xticks(np.arange(-50, 50, 10.0))
plt.show()

# =========== Part 2: Regularized Linear Regression Cost =============
theta = np.ones((2,1))
J = linearRegCostFunction(theta, np.column_stack((np.ones((m,1)), X)), y, 1)
print('Cost at theta = [1 ; 1] - (this value should be about 303.993192)\n', J)

# =========== Part 3: Regularized Linear Regression Gradient =============
J, grad = linearRegCostFunction(theta, np.column_stack((np.ones((m,1)), X)), y, 1, True)
print('Gradient at theta = [1 ; 1] - (this value should be about [-15.303016; 598.250744])\n', grad)

# =========== Part 4: Train Linear Regression =============
_lambda = 0
result = trainLinearReg(np.column_stack((np.ones((m,1)), X)), y, _lambda)
plt.plot(X, y, marker='x', linestyle='None')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlabel('Change in water level (x)')
plt.xticks(np.arange(-50, 50, 10.0))
plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), result.x))
plt.show()

# =========== Part 5: Learning Curve for Linear Regression =============
_lambda = 0
error_train, error_val = learningCurve(np.column_stack((np.ones((m,1)), X)), y, \
                            np.column_stack((np.ones((Xval.shape[0],1)), Xval)), yval, _lambda)
plt.plot(range(0, m), error_train, label="Training Error")
plt.plot(range(0, m), error_val, label="Validation Error")
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()
print('Training Examples\tTrain Error\tCross Validation Error\n')
for i in  range(0, m):
    print('{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

# =========== Part 6: Feature Mapping for Polynomial Regression =============
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

# =========== Part 7: Learning Curve for Polynomial Regression =============
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
plt.show()

plt.figure(2)
error_train, error_val = learningCurve(x_poly, y, x_poly_val, yval, _lambda)
p1, p2 = plt.plot(range(1,m+1), error_train, range(1,m+1), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(_lambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 50])
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.show()
print('Polynomial Regression (lambda = {:f})\n\n'.format(_lambda))
print('Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(0, m):
    print('{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

# =========== Part 8: Validation for Selecting Lambda =============
plt.close()
lambda_vec, error_train, error_val, theta_train = validationCurve(x_poly, y, x_poly_val, yval)
p1, p2 = plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()
#print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(0, len(lambda_vec)):
    print('{:f}\t\t{:f}\t{:f}\n'.format(lambda_vec[i], float(error_train[i]), float(error_val[i])))
    #print("theta at ", i,": ",theta_train[i])

# Compute test set error
theta = theta_train[np.argmin(error_val)]
J = linearRegCostFunction(theta, x_poly_test, ytest, 0)
print("Test set error: ", J)

# Plot learning curves with randomly selected examples
_lambda = 0.01
iteration = 50
m, n = x_poly.shape
# initialize error matrices
random_error_train = np.zeros((m, iteration))
random_error_val   = np.zeros((m, iteration))

for i in range(0, m):
    for k in range(iteration):
        random_train = np.random.permutation(x_poly.shape[0])[0:i+1]
        random_x_poly = x_poly[random_train, :].reshape(i+1, n)
        random_y = y[random_train].reshape(i+1, 1)

        random_val = np.random.permutation(x_poly_val.shape[0])[0:i+1]
        random_x_poly_val = x_poly_val[random_val, :].reshape(i+1, n)
        random_yval = yval[random_val].reshape(i+1, 1)

        result = trainLinearReg(random_x_poly, random_y, _lambda)
        theta = result.x
        random_error_train[i,k] = linearRegCostFunction(theta, random_x_poly, random_y, 0)
        random_error_val[i,k]   = linearRegCostFunction(theta, random_x_poly_val, random_yval, 0)

error_train = np.mean(random_error_train, axis=1)
error_val   = np.mean(random_error_val, axis=1)

plt.close()
p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(_lambda))
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()

print('Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('{:d}\t\t{:f}\t{:f}\n'.format(i+1, error_train[i], error_val[i]))
