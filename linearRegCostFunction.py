import numpy as np


def linear_reg_cost_function(theta, x, y, lmd):
    # Number of training examples
    m = y.size

    # Inintialize cost and gradient
    cost = 0
    grad = np.zeros(theta.shape)

    # Calculate error term but computing the difference between predicted and actual values
    error = np.dot(x, theta) - y

    # Compute the regularized cost function
    cost = (np.sum(error ** 2) / (2 * m)) + (np.sum(theta[1:] ** 2) * lmd / (2 * m))

    # Compute the regularization term for the gradient
    reg_term = theta * (lmd / m)
    reg_term[0] = 0 # Exclude regularization term from bias

    # Compute the gradient with regularization
    grad = np.dot(x.T, error) / m + reg_term

    return cost, grad
