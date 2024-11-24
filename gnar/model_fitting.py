import numpy as np

def format_X_y(data, p, s_vec):
    """
    Format the data to fit the GNAR model

    Params:
        data: np.array. Time series and neighbour sums. Shape (m, n, max(s_vec) + 1)
        p: int. Number of lags
        s_vec: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (m - p, n, p + sum(s)). The lagged node values are the first p columns, followed by the neighbour sums for each lag.
        y: np.array. Target matrix. Shape (m - p, n)
    """
    m, n, _ = np.shape(data)
    # Create the design matrix, which will be an array of shape (p + sum(s), m - p, n)
    X = np.zeros([m - p, n, p + sum(s_vec)])
    # Create the target matrix, which will be an array of shape (m - p, n)
    y = data[p:, :, 0]
    # Fill the design matrix with the lagged values and neighbour sums
    for i in range(p):
        X[:, :, i] = data[p - i - 1 : m - i - 1, :, 0]
        X[:, :, p + np.sum(s_vec[:i]).astype(int) : p + np.sum(s_vec[:i+1]).astype(int)] = data[p - i - 1 : m - i - 1, :, 1 : s_vec[i] + 1]
    # Return the design and target matrices
    return X, y

def gnar_lr(X, y, p, s_vec, model_type):
    """
    Fit a GNAR model using multiple linear regression

    Params:
        X: np.array. Design matrix. Shape (m, n, p + sum(s))
        y: np.array. Target matrix. Shape (m, n)
        p: int. Number of lags
        s_vec: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)
        model_type: str. Type of GNAR model to fit. One of "global", "standard" or "local"

    Returns:
        coeffs_mat: np.array. Coefficients matrix. Shape (n, p + sum(s))
    """
    if model_type == "global":
        return global_gnar_lr(X, y)
    elif model_type == "standard":
        return standard_gnar_lr(X, y, p, s_vec)
    elif model_type == "local":
        return local_gnar_lr(X, y)
    else:
        raise ValueError("Invalid model type")

def global_gnar_lr(X, y):
    """
    Fit a global-alpha GNAR model using multiple linear regression. In the global-alpha GNAR model, the coefficients are the same for all nodes.

    Parameters:
        X (ndarray): The design matrix. Shape (m, n, k), where m in the number of observation, n is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (m, n)

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (n, k)
    """
    # Shape of the data
    m, n, k = np.shape(X)
    # Stack the design matrix and target
    design_matrix = np.transpose(X, (1, 0, 2)).reshape(n * m, k)
    target = y.T.reshape(-1,1)
    # Compute the coefficients using least squares regression
    coeffs = np.zeros(k)
    valid_cols = np.sum(np.abs(design_matrix), axis=0) > 0
    coeffs[valid_cols] = np.linalg.lstsq(design_matrix[:, valid_cols], target, rcond=None)[0].flatten()
    # Remap the coefficients to the original shape
    coeffs_mat = np.repeat(coeffs.reshape(1, k), n, axis=0)
    # Return the coefficients matrix
    return coeffs_mat


def standard_gnar_lr(X, y, p, s_vec):
    """
    Fit a standard GNAR model using multiple linear regression. In the standard GNAR model, the alpha coefficients are different for each node, while the beta coeffients are the same.

    Parameters:
        X (ndarray): The design matrix. Shape (m, n, k), where m in the number of observation, n is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (m, n)
        p (int): The number of lags.
        s_vec (ndarray): An array containing the maximum stage of neighbour dependence for each lag.

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (n, k)
    """
    # Shape of the data
    m, n, k = np.shape(X)
    # Initialise the design matrix and target
    design_matrix = np.zeros([m * n, p * n])
    target = y.T.reshape(-1,1)
    # Fill the design matrix with the alpha coefficient features
    for i in range(n):
        design_matrix[i * m : (i + 1) * m, i :: n] = X[:, i, :p]
    # Add the beta coeffient features to the design matrix
    design_matrix = np.hstack([design_matrix, np.transpose(X[:, :, p:], (1, 0, 2)).reshape(n * m, np.sum(s_vec))])
    # Compute the coefficients using least squares regression
    coeffs = np.zeros(p * n + np.sum(s_vec))
    valid_cols = np.sum(np.abs(design_matrix), axis=0) > 0
    coeffs[valid_cols] = np.linalg.lstsq(design_matrix[:, valid_cols], target, rcond=None)[0].flatten()
    # Remap the coefficients to the original shape
    coeffs_mat = coeffs[0 : n * p].reshape(p, n).T
    coeffs_mat = np.hstack([coeffs_mat, np.repeat(coeffs[n * p :].reshape(1, -1), n, axis=0)])
    # Return the coefficients matrix
    return coeffs_mat


def local_gnar_lr(X, y):
    """
    Fit a local-beta GNAR model using multiple linear regression. In the local-beta GNAR model, the alphaand beta coefficients are different for each node.

    Parameters:
        X (ndarray): The design matrix. Shape (m, n, k), where m in the number of observation, n is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (m, n)
        p (int): The number of lags.
        s_vec (ndarray): An array containing the maximum stage of neighbour dependence for each lag.

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (n, k)
    """
    # Shape of the data
    m, n, k = np.shape(X)
    # Compute the coefficients for each node using least squares regression
    coeff_mat = np.zeros((n, k))
    for i in range(n):
        X_i = X[:, i, :]
        valid_cols = np.sum(np.abs(X_i), axis=0) > 0
        coeff_mat[i, valid_cols > 0] = np.linalg.lstsq(X_i[:, valid_cols > 0], y[:, i], rcond=None)[0].flatten()
    # Return the coefficients matrix
    return coeff_mat


def constrained_multiple_lr(X, y, constraints):
    """
    Fit a multiple linear regression model with the constraint that a set of coefficients are equal among all fits.

    Parameters:
        X (ndarray): The design matrix. Shape (m, n, k), where m in the number of observation, n is the number of nodes and k is the number of features.
        y (ndarray): The target matrix. Shape (m, n)
        constraints (ndarray): The coefficients which are not allowed to vary among nodes, with 1s denoting the ones that are fixed and 0s the ones that are allowed to vary. Shape (k,)

    Returns:
        coeffs_mat (ndarray): The coefficients matrix. Shape (n, k)
    """
    # Shape of the data
    m, n, k = np.shape(X)
    # Number of constrained and free coefficients
    n_constrained = np.sum(constraints)
    n_free = k - n_constrained
    design_k = n_constrained + n_free * n
    # Initialise the design matrix and target
    design_matrix = np.zeros([m * n, design_k])
    # Stack the columns of y on top of each other to create a single target vector
    target = y.T.reshape(-1,1)
    # Initialise the counter to keep track of the column index
    counter = 0
    for i in range(k):
        # If the coefficient is constrained, add it to the design matrix as a single column
        if constraints[i] == 1:
            design_matrix[:, counter] = X[:, :, i].T.reshape(-1)
            counter += 1
        # If the coefficient is free, add it to the design matrix as n columns in the corresponding locations
        else:
            for j in range(n):
                design_matrix[j * m : (j + 1) * m, counter] = X[:, j, i]
                counter += 1
                
    # Only compute coefficients for columns that have at least one non-zero value
    valid_cols = np.sum(np.abs(design_matrix), axis=0) > 0
    coeffs = np.zeros(design_k)
    # Compute the coefficients using least squares regression
    coeffs[valid_cols] = np.linalg.lstsq(design_matrix[:, valid_cols], target, rcond=None)[0].flatten()
    # Remap the coefficients to the original shape
    coeffs_mat = np.zeros([n, k])
    counter = 0
    for i in range(k):
        # If the coefficient is constrained, then it only needs to be added once as a single column
        if constraints[i] == 1:
            coeffs_mat[:, i] = coeffs[counter]
            counter += 1
        # If the coefficient is free, then it needs to be added n times in the corresponding location
        else:
            coeffs_mat[:, i] = coeffs[counter : counter + n]
            counter += n
    # Return the coefficients matrix
    return coeffs_mat    

    

