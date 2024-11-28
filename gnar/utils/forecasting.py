import numpy as np

def format_X(data, p, s):
    """
    Format the data to compute forecasts using the GNAR model

    Params:
        data: np.array. Time series and neighbour sums. Shape (n, d, 1 + max(s))
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (n - p + 1, d, p + sum(s))
        lagged_vals: np.array. Lagged values of neighbour sums. Shape (n - p + 1, d, p * (max(s)))
    """
    # Data shapes
    n, d, _ = np.shape(data)
    r = np.max(s)
    # Create the design matrix, which will be an array of shape (p + sum(s), n - p, d)
    X = np.zeros([n - p + 1, d, p + sum(s)])
    # Create the lagged values array, which will be an array of shape (n - p + 1, d, p * (1 + max(s))). This stores the lagged node values and neighbour sums
    lagged_vals = np.zeros([n - p + 1, d, p * r])
    # Fill the design matrix and the lagged values with the observations and neighbour sums
    for i in range(p):
        lagged_vals[:, :, i * r : (i + 1) * r] = data[p - i - 1 : n - i, :, 1:]
        X[:, :, i] = data[p - i - 1 : n - i, :, 0]
        X[:, :, p + np.sum(s[:i]).astype(int) : p + np.sum(s[:i + 1]).astype(int)] = data[p - i - 1 : n - i, :, 1 : s[i] + 1]
    # Return the design matrix and the lagged values
    return X, lagged_vals

def update_X(X, preds, lagged_vals, p, s):
    """
    Update the design matrix with new lagged values and neighbour sums

    Params:
        X: np.array. Design matrix. Shape (n, d, p + sum(s))
        preds: np.array. Predicted values. Shape (n, d)
        lagged_vals: np.array. Lagged values and neighbour sums. Shape (n, d, p * (1 + max(s)))
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        np.array. Design matrix. Shape (n, d, p + sum(s))
    """
    # Data shapes
    n, d, _ = np.shape(X)
    r = np.max(s)
    # Update the lagged time series observations
    X[:, :, 1 : p] = X[:, :, : p - 1]
    X[:, :, 0] = preds
    # Update the design matrix with the lagged neighbour sums
    for i in range(p):
        X[:, :, p + np.sum(s[:i]).astype(int) : p + np.sum(s[:i+1]).astype(int)] = lagged_vals[:, :, i * r : i * r + s[i]]
    return X