import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from loadMovieList import loadMovieList
from normalizeRatings import normalizeRatings

#main block
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# =============== Part 1: Loading movie ratings dataset ================

data = loadmat('data/ex8_movies.mat')
Y = data['Y']
R = data['R']

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): ', np.mean(Y[0, R[0, :]==1]))

#plt.imshow(Y, aspect='auto')
#plt.ylabel('Movies')
#plt.xlabel('Users')
#plt.show()

# ============ Part 2: Collaborative Filtering Cost Function ===========

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data2 = loadmat ('data/ex8_movieParams.mat')
X = data2['X']
Theta = data2['Theta']
num_users = data2['num_users']
num_movies = data2['num_movies']
num_features = data2['num_features']

num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]
#print(X)
#print(Theta)
params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
print('Cost at loaded parameters: (this value should be about 22.22)', J)

# ============== Part 3: Collaborative Filtering Gradient ==============

print('Checking Gradients (without regularization)...')
checkCostFunction()

# ========= Part 4: Collaborative Filtering Cost Regularization ========

J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)         
print('Cost at loaded parameters (lambda = 1.5): (this value should be about 31.34)', J)

# ======= Part 5: Collaborative Filtering Gradient Regularization ======

print('Checking Gradients (with regularization)...')
checkCostFunction(1.5)

# ============== Part 6: Entering ratings for a new user ===============

movieList = loadMovieList()

#  Initialize my ratings
my_ratings = np.zeros((1682, 1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('New user ratings: \n')
for i in range (len(my_ratings)):
    if my_ratings[i] > 0: 
        print('Rated ', my_ratings[i],' for ', movieList[i])

# ================== Part 7: Learning Movie Ratings ====================

data3 = loadmat('data/ex8_movies.mat')
Y = data3['Y']
R = data3['R']

#  Add our own ratings to the data matrix
Y = np.column_stack((my_ratings, Y))
R = np.column_stack(((my_ratings != 0), R))

#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

# Set Regularization
_lambda = 10
fargs = (Ynorm, R, num_users, num_movies, num_features, _lambda)
result = minimize(cofiCostFunc, x0=initial_parameters, args=fargs, method="L-BFGS-B", options={'disp': True, 'maxiter': 100})
params = result.x                

# Unfold the returned theta back into U and W
X = params[0:num_movies*num_features].reshape(num_movies, num_features, order='F')
Theta = params[num_movies*num_features:].reshape(num_users, num_features, order='F')

print('Recommender system learning completed.')

# ================== Part 8: Recommendation for you ====================

p = np.dot(X, Theta.T)
my_predictions = p[:,0] + Ymean.flatten()
print(my_predictions)
movieList = loadMovieList()

ix = np.argsort(my_predictions)[::-1]
print('Top recommendations for you:')
for i in range(10):
    j = ix[i]
    print('Predicting rating ', my_predictions[j],' for movie ', movieList[j])

print('Original ratings provided:')
for i in range (len(my_ratings)):
    if my_ratings[i] > 0: 
        print('Rated ', my_ratings[i],' for ', movieList[i])
