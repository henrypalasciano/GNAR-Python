import numpy as np

def format_X(data, p, s_vec):
    """
    Format the data to compute forecasts using the GNAR model

    Params:
        data: np.array. Time series and neighbour sums. Shape (m, n, 1 + max(s_vec))
        p: int. Number of lags
        s_vec: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (m - p + 1, n, p + sum(s))
        lagged_vals: np.array. Lagged values of neighbour sums. Shape (m - p + 1, n, p * (max(s_vec)))
    """
    # Data shapes
    m, n, _ = np.shape(data)
    r = np.max(s_vec)
    # Create the design matrix, which will be an array of shape (p + sum(s), m - p, n)
    X = np.zeros([m - p + 1, n, p + sum(s_vec)])
    # Create the lagged values array, which will be an array of shape (m - p + 1, n, p * (1 + max(s))). This stores the lagged node values and neighbour sums
    lagged_vals = np.zeros([m - p + 1, n, p * r])
    # Fill the design matrix and the lagged values with the observations and neighbour sums
    for i in range(p):
        lagged_vals[:, :, i * r : (i + 1) * r] = data[p - i - 1 : m - i, :, 1:]
        X[:, :, i] = data[p - i - 1 : m - i, :, 0]
        X[:, :, p + np.sum(s_vec[:i]).astype(int) : p + np.sum(s_vec[:i+1]).astype(int)] = data[p - i - 1 : m - i, :, 1 : s_vec[i] + 1]
    # Return the design matrix and the lagged values
    return X, lagged_vals

def update_X(X, preds, lagged_vals, p, s_vec):
    """
    Update the design matrix with new lagged values and neighbour sums

    Params:
        X: np.array. Design matrix. Shape (m, n, p + sum(s))
        preds: np.array. Predicted values. Shape (m, n)
        lagged_vals: np.array. Lagged values and neighbour sums. Shape (m, n, p * (1 + max(s_vec)))
        p: int. Number of lags
        s_vec: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        new_X: np.array. Design matrix. Shape (m, n, p + sum(s))
    """
    # Data shapes
    m, n, _ = np.shape(X)
    r = np.max(s_vec)
    # Update the lagged time series observations
    X[:, :, 1 : p] = X[:, :, : p - 1]
    X[:, :, 0] = preds
    # Update the design matrix with the lagged neighbour sums
    for i in range(p):
        X[:, :, p + np.sum(s_vec[:i]).astype(int) : p + np.sum(s_vec[:i+1]).astype(int)] = lagged_vals[:, :, i * r : i * r + s_vec[i]]
    return X