import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def learning_curve(X, y, Xval, yval, lmd):
    # Number of training examples
    m = X.shape[0]

    # Initialize errors in training and validation
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        # Select the first i examples from the training set
        x_i = X[:i+1]
        y_i = y[:i+1]

        # Train linear regression model using regularization
        theta = tlr.train_linear_reg(x_i, y_i, lmd)

        # Compute training and validation errors without regularization (lmd = 0)
        error_train[i] = lrcf.linear_reg_cost_function(theta, x_i, y_i, 0)[0]
        error_val[i] = lrcf.linear_reg_cost_function(theta, Xval, yval, 0)[0]

    return error_train, error_val
