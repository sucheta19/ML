import numpy as np
import os

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


data =  np.loadtxt(os.path.join('/home/sucheta/Desktop/ML/Data', 'ex1data1.txt'), delimiter = ',')

X, y = data[:,0],data[:,1]

n = 1   #number of features


m = y.size      #number of training examples
theta = np.zeros(n+1)

def plotdata(x,y):
    pyplot.plot(x,y,'ro', ms = 5, mec = 'k')
    pyplot.xlabel('Profit')
    pyplot.ylabel('Population')

plotdata(X,y)

#adding X0 feature to design matrix
X = np.stack([np.ones(m),X], axis = 1)


def computeCost(X, y, theta):
    J = 0
    for output, feature_vector  in zip(y, X):
        h = np.dot(np.transpose(theta), feature_vector)
        J = J + (h - output)**2
    J = J/(2*m)
    return J


def computeDer(X, y, theta, t):
    J = 0
    for output, feature_vector  in zip(y, X):
        h = np.dot(np.transpose(theta), feature_vector)
        J = J + (h - output)*feature_vector[t]
    J = J/(m)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    theta = theta.copy()
    temp = theta.copy()
    J_history = [] # Use a python list to save cost in every iteration
    t = 0
    for i in range(num_iters):
        for j in range(temp.shape[0]):
            temp[j] = temp[j] - alpha * (computeDer(X ,y, theta, j))
        theta = temp
        # save the cost J in every iteration
        #J_history.append(computeCost(X, y, theta))

    return theta, J_history


theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')


# For visualizing

#plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);


# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = pyplot.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pass
