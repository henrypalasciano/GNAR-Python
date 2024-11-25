import numpy as np

def shift_X(X, sim, ns, p, s_vec):
    """
    Shift the design matrix with the new simulated values and neighbour sums

    Params:
        X: np.array. Design matrix. Shape (n, p + sum(s))
        sim: np.array. Simulated values. Shape (n,)
        ns: np.array. Neighbour sums. Shape (p, n, max(s))
        p: int. Number of lags
        s_vec: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (n, p + sum(s))
    """
    # Data shapes
    n, _ = np.shape(X)
    r = np.max(s_vec)
    # Update the lagged time series observations
    X[:, 1 : p] = X[:, : p - 1]
    X[:, 0] = sim
    # Update the design matrix with the lagged neighbour sums
    for i in range(p):
        X[:, p + np.sum(s_vec[:i]).astype(int) : p + np.sum(s_vec[:i+1]).astype(int)] = ns[:, i * r : i * r + s_vec[i]]
    return X