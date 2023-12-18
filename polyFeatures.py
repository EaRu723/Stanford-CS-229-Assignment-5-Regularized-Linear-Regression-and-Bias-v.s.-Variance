import numpy as np

# Generate polunomial features up to degree p for a given input array X
# X: Input array of shape (m,), where m is the number of samples
# p: Degree of the polynomial

# Returs,: 
# X_poly: Polynomial feature matrix of shape (m,p) where each column represents a polynomial feature
def poly_features(X, p):
    # Initialize output matrix
    X_poly = np.zeros((X.size, p))

    # Generate the powers of X up to degree p
    P = np.arange(1, p+1)

    # Compute the polynomial features
    X_poly = X.reshape((X.size, 1)) ** P

    return X_poly