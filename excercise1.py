# Programming Exercise 1: Linear Regression : warmUpExercise

import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

def warmUpExercise():
    """
    Example function in Python which computes the identity matrix.

    Returns
    -------
    A : array_like
        The 5x5 identity matrix.

    Instructions
    ------------
    Return the 5x5 identity matrix.
    """
    # ======== YOUR CODE HERE ======
    A = np.identity(5)
    print(A)

    # ==============================
    return A

# Read comma separated data
data = np.loadtxt(os.path.join('/home/sucheta/Desktop/ML/Data', 'ex1data1.txt'), delimiter=',') #home/sucheta/Desktop/ML/Data/ex1data1.txt
X, y = data[:, 0], data[:, 1]

m = y.size  # number of training examples

#2 Linear regression with one variable

#2.1 Plotting the Data
def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """


    # ====================== YOUR CODE HERE =======================

    pyplot.plot(x,y,'ro',mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')
    #pyplot.show()

plotData(X,y)

# Add a column of ones to X. The numpy function stack joins arrays along a given axis.
# The first axis (axis=0) refers to rows (training examples)
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(m), X], axis=1)

def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta.
    You should set J to the cost.
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



J = computeCost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')

def diffcostfn(theta,X,y):
    global H
    m=y.size
    H=X.dot(theta.T)
    dtheta0=0
    dtheta1=0
    for i in range(m):
        dtheta0+=(H[i]-y[i])/(m)
        dtheta1+=((H[i]-y[i])*X[i][1])/(m)
    #print("dtheta1",dtheta1)
    return dtheta0,dtheta1




def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

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
          J0,J1=diffcostfn(theta,X,y)
          temp0=theta[0]-alpha*J0
          temp1=theta[1]-alpha*J1
          #print("temp1",temp1)
          #print("Q1",theta[1])
          theta[0]=temp0
          theta[1]=temp1
        # =====================================================================
        # save the cost J in every iteration
          J_history.append(computeCost(X, y, theta))

    return theta, J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')



# plot the linear fit
#plotData(X[:,1],y)
pyplot.plot(X[:,1],H,'b-')
pyplot.legend(['Training data', 'Linear regression'])
