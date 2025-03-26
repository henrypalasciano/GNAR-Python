import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix

def format_X_y(data, p, s):
    """
    Format the data to fit the GNAR model

    Params:
        data: np.array. Time series and neighbour sums. Shape (n, d, 1 + r), where r = max(s)
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (n - p, d, p + sum(s)). The lagged node values are the first p columns, followed by the neighbour sums for each lag.
        y: np.array. Target matrix. Shape (n - p, d)
    """
    n, d, k = np.shape(data)
    # Initialise an empty design matrix, an array of shape (p + sum(s), n - p, d)
    X = np.zeros([n - p, d, p + sum(s)])
    # Fill the design matrix
    for i in range(p):
        # Lag j alpha coefficient data is placed in the j-th column of the design matrix
        X[:, :, i] = data[p - i - 1 : n - i - 1, :, 0]
        # Lag j beta coefficient data is placed in the (p + sum(s[:j]))-th to (p + sum(s[:j+1]))-th columns of the design matrix
        start = p + np.sum(s[:i]).astype(int)
        end = start + s[i]
        X[:, :, start:end] = data[p - i - 1 : n - i - 1, :, 1 : s[i] + 1]
    # Target matrix, an array of shape (n - p, d)
    y = data[p:, :, 0]
    return X, y


def gnar_lr(data, p, s, model_type):
    """
    Fit a GNAR model using multiple linear regression

    Params:
        X: np.array. Design matrix. Shape (n, d, p + sum(s))
        y: np.array. Target matrix. Shape (n, d)
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)
        model_type: str. Type of GNAR model to fit. One of "global", "standard" or "local"

    Returns:
        coeffs_mat: np.array. Coefficients matrix. Shape (d, p + sum(s))
    """
    # Format the data
    X,y = format_X_y(data, p, s)
    # Shape of the data
    n, d, k = np.shape(X)
    
    if model_type == "global":
        # Stack the design matrix and target
        design_matrix = np.transpose(X, (1, 0, 2)).reshape(d * n, k)
        target = y.T.reshape(-1,1)

        # Compute the coefficients using least squares regression
        coeffs = np.zeros(k)
        valid_cols = np.any(design_matrix != 0, axis=0)
        coeffs[valid_cols] = lstsq(design_matrix[:, valid_cols], target)[0].flatten()
        # Remap the coefficients to the original shape
        coeffs_mat = np.tile(coeffs, (d, 1))
    
    elif model_type == "standard":
        # Initialise the design matrix and target vector
        design_matrix = np.zeros([n * d, p * d])
        target = y.T.reshape(-1,1)
        # Fill the design matrix with the alpha coefficient features
        for i in range(d):
            design_matrix[i * n : (i + 1) * n, i :: d] = X[:, i, :p]
        # Add the beta coeffient features to the design matrix
        design_matrix = np.hstack([design_matrix, np.transpose(X[:, :, p:], (1, 0, 2)).reshape(d * n, np.sum(s))])
        # Compute the coefficients using least squares regression
        coeffs = np.zeros(p * d + np.sum(s))
        valid_cols = np.any(design_matrix != 0, axis=0)
        # Use a sparse representation for efficient computation
        sparse_design_matrix = csr_matrix(design_matrix[:, valid_cols])
        coeffs[valid_cols] = lsqr(sparse_design_matrix, target)[0].flatten()
        # Remap the coefficients to the original shape
        coeffs_mat = coeffs[0 : d * p].reshape(p, d).T
        coeffs_mat = np.hstack([coeffs_mat, np.repeat(coeffs[d * p :].reshape(1, -1), d, axis=0)])
    
    elif model_type == "local":
        # Compute the coefficients for each node separately
        coeffs_mat = np.zeros((d, k))
        for i in range(d):
            X_i = X[:, i, :]
            valid_cols = np.any(X_i != 0, axis=0)
            coeffs_mat[i, valid_cols > 0] = lstsq(X_i[:, valid_cols > 0], y[:, i])[0].flatten()
    
    # Compute the covariance matrix of the residuals
    res = np.sum(X * coeffs_mat, axis=2) - y
    sigma_2 = res.T @ res / (n - p)

    return coeffs_mat, sigma_2