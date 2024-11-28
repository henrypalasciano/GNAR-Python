import numpy as np

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
    # Create the design matrix, which will be an array of shape (p + sum(s), n - p, d)
    X = np.zeros([n - p, d, p + sum(s)])
    # Create the target matrix, which will be an array of shape (n - p, d)
    y = data[p:, :, 0]
    # Fill the design matrix with the lagged values and neighbour sums
    for i in range(p):
        X[:, :, i] = data[p - i - 1 : n - i - 1, :, 0]
        X[:, :, p + np.sum(s[:i]).astype(int) : p + np.sum(s[:i+1]).astype(int)] = data[p - i - 1 : n - i - 1, :, 1 : s[i] + 1]
    # Return the design and target matrices
    return X, y

def gnar_lr(X, y, p, s, model_type):
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
    if model_type == "global":
        return global_gnar_lr(X, y)
    elif model_type == "standard":
        return standard_gnar_lr(X, y, p, s)
    elif model_type == "local":
        return local_gnar_lr(X, y)
    else:
        raise ValueError("Invalid model type")

def global_gnar_lr(X, y):
    """
    Fit a global-alpha GNAR model using multiple linear regression. In the global-alpha GNAR model, the coefficients are the same for all nodes.

    Parameters:
        X (ndarray): The design matrix. Shape (n, d, k), where n in the number of observation, d is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (n, d)

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (d, k)
    """
    # Shape of the data
    n, d, k = np.shape(X)
    # Stack the design matrix and target
    design_matrix = np.transpose(X, (1, 0, 2)).reshape(d * n, k)
    target = y.T.reshape(-1,1)
    # Compute the coefficients using least squares regression
    coeffs = np.zeros(k)
    valid_cols = np.sum(np.abs(design_matrix), axis=0) > 0
    coeffs[valid_cols] = np.linalg.lstsq(design_matrix[:, valid_cols], target, rcond=None)[0].flatten()
    # Remap the coefficients to the original shape
    coeffs_mat = np.repeat(coeffs.reshape(1, k), d, axis=0)
    # Return the coefficients matrix
    return coeffs_mat


def standard_gnar_lr(X, y, p, s):
    """
    Fit a standard GNAR model using multiple linear regression. In the standard GNAR model, the alpha coefficients are different for each node, while the beta coeffients are the same.

    Parameters:
        X (ndarray): The design matrix. Shape (n, d, k), where n in the number of observation, d is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (n, d)
        p (int): The number of lags.
        s (ndarray): An array containing the maximum stage of neighbour dependence for each lag.

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (d, k)
    """
    # Shape of the data
    n, d, k = np.shape(X)
    # Initialise the design matrix and target
    design_matrix = np.zeros([n * d, p * d])
    target = y.T.reshape(-1,1)
    # Fill the design matrix with the alpha coefficient features
    for i in range(d):
        design_matrix[i * n : (i + 1) * n, i :: d] = X[:, i, :p]
    # Add the beta coeffient features to the design matrix
    design_matrix = np.hstack([design_matrix, np.transpose(X[:, :, p:], (1, 0, 2)).reshape(d * n, np.sum(s))])
    # Compute the coefficients using least squares regression
    coeffs = np.zeros(p * d + np.sum(s))
    valid_cols = np.sum(np.abs(design_matrix), axis=0) > 0
    coeffs[valid_cols] = np.linalg.lstsq(design_matrix[:, valid_cols], target, rcond=None)[0].flatten()
    # Remap the coefficients to the original shape
    coeffs_mat = coeffs[0 : d * p].reshape(p, d).T
    coeffs_mat = np.hstack([coeffs_mat, np.repeat(coeffs[d * p :].reshape(1, -1), d, axis=0)])
    # Return the coefficients matrix
    return coeffs_mat


def local_gnar_lr(X, y):
    """
    Fit a local-beta GNAR model using multiple linear regression. In the local-beta GNAR model, the alphaand beta coefficients are different for each node.

    Parameters:
        X (ndarray): The design matrix. Shape (n, d, k), where n in the number of observation, d is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (n, d)
        p (int): The number of lags.
        s (ndarray): An array containing the maximum stage of neighbour dependence for each lag.

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (d, k)
    """
    # Shape of the data
    n, d, k = np.shape(X)
    # Compute the coefficients for each node using least squares regression
    coeff_mat = np.zeros((d, k))
    for i in range(d):
        X_i = X[:, i, :]
        valid_cols = np.sum(np.abs(X_i), axis=0) > 0
        coeff_mat[i, valid_cols > 0] = np.linalg.lstsq(X_i[:, valid_cols > 0], y[:, i], rcond=None)[0].flatten()
    # Return the coefficients matrix
    return coeff_mat