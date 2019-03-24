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
    pyplot.show()

plotData(X, y)
