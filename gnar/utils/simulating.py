import numpy as np

def shift_X(X, sim, ns, p, s):
    """
    Shift the design matrix with the new simulated values and neighbour sums

    Params:
        X: np.array. Design matrix. Shape (d, p + sum(s))
        sim: np.array. Simulated values. Shape (d,)
        ns: np.array. Neighbour sums. Shape (p, d, max(s))
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (d, p + sum(s))
    """
    # Data shapes
    d, _ = np.shape(X)
    r = np.max(s)
    # Update the lagged time series observations
    X[:, 1 : p] = X[:, : p - 1]
    X[:, 0] = sim
    # Update the design matrix with the lagged neighbour sums
    for i in range(p):
        X[:, p + np.sum(s[:i]).astype(int) : p + np.sum(s[:i+1]).astype(int)] = ns[:, i * r : i * r + s[i]]
    return X

def generate_noise(sigma_2, n, d):
    """
    Generate noise for the GNAR model. The noise is assumed to be Gaussian with mean 0 and variance (or covariance) sigma_2.

    Params:
        sigma_2: float, int, np.array. The variance or covariance matrix.
        n: int. Number of observations.
        d: int. Number of nodes.
    
    Returns:
        np.array. Noise matrix. Shape (n, d)
    """
    if isinstance(sigma_2, (float, int)):
        return np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(n, d))
    elif sigma_2.ndim == 1 or sigma_2.shape[0] == 1:
        return np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(n, d))
    return np.random.multivariate_normal(mean=np.zeros(d), cov=sigma_2, size=n)