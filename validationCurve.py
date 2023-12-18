import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def validation_curve(X, y, Xval, yval):
    # Selected values of lambda
    lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # Initialize arrays to store training and validation errors for each lambda
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

    # Loop throuhg values of lambda
    for i in range(lambda_vec.size):
        lmd = lambda_vec[i]
        # Train linear regression model with regularization
        theta = tlr.train_linear_reg(X, y, lmd)

        # Calculate training error without regularization
        error_train[i] = lrcf.linear_reg_cost_function(theta, X, y, lmd)[0]

        # Calculate validation error without regularization
        error_val[i] = lrcf.linear_reg_cost_function(theta, Xval, yval, lmd)[0]

    return lambda_vec, error_train, error_val
