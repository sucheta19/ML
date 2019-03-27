import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline


# Load data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:,:2]
y = data[:, 2]
#m = y.size
m,n= X.shape
print(m)
"""
# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
"""

def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    global m,n
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    for i in range(n):
        mu[i]=np.mean(X[:,i])
        sigma[i]=np.std(X[:,i])
        X_norm[:,i]=(X[:,i]-mu[i])/sigma[i]
    # ================================================================
    return X_norm, mu, sigma



# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
print('X_norm',X)

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    Returns
    -------
    J : float
        The value of the cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    global H
    # initialize some useful values
    m = y.size  # number of training examples
    H=X.dot(theta.T)
    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE =====================
    for i in range(m):
        J+=((H[i]-y[i])**2)/(2*m)

    # ===========================================================
    return J

def diffcostfn(theta,X,y):
    global H,m,n

    H=X.dot(theta.T)
    dtheta=np.zeroes((1,n+1))
    for j in range(n+1):
        for i in range(m):
            dtheta[j]+=((H[i]-y[i])*X[i][1])/(m)
            #dtheta1+=((H[i]-y[i])*X[i][1])/(m)
        #print("dtheta1",dtheta1)
    return dtheta
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    alpha : float
        The learning rate for gradient descent.

    num_iters : int
        The number of iterations to run gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    global H
    # Initialize some useful values
    #temp0=np.zeroes
    m = y.shape[0]  # number of training examples
    #H=X.dot(theta.T)
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()

    J_history = [] # Use a python list to save cost in every iteration

    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
          Jth=diffcostfn(theta,X,y)
          temp=np.zeroes((1,n+1))
          for d in range(n+1):
              temp[i]=theta[i]-alpha*Jth[i]
          for d in range(n+1):
              theta[i]=temp[i]

        # =====================================================================
        # save the cost J in every iteration
          J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history
