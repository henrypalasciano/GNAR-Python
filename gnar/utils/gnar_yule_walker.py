import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

from gnar.utils.gnar_linear_regression import format_X_y

def estimate_covariance_mats(ts, p):
    """
    Estimate the covariance matrices for tau = 0, 1, ..., p.
    
    Params:
        ts: np.ndarray of shape (n, d). Time series data.
        p: int. Number of lags.
    """
    n, d = ts.shape
    cov_mats = np.zeros([p + 1, d, d])
    # Compute the covariance matrix for tau = 0
    cov_mats[0] = ts.T @ ts / (n - 1)
    # Compute the covariance matrices for tau = 1, ..., p
    for tau in range(1, p + 1):
        cov_mats[tau] = ts[tau:].T @ ts[:n - tau] / (n - tau - 1)
    return cov_mats

def network_covariance_mats(autocov_mats, ns_mats, p):
    """
    Given the autocovariance matrices for tau = 0, 1, ..., p and the neighbour sum matrices, construct the network autocovariance matrices for each node, lag and stage of neighbour dependence.

    Params:
        autocov_mats: np.ndarray of shape (p + 1, d, d). Covariance matrices for tau = 0, 1, ..., p.
        ns_mats: np.ndarray of shape (max(s), d, d). Neighbour sum matrices for r = 1, ..., max(s).
        p: int. Number of lags.
    
    Returns:
        np.ndarray of shape (d, 2 * p + 1, r + 1, r + 1). Network autocovariance matrices for each node, lag and stage of neighbour dependence.
    """
    r, d, _ = ns_mats.shape
    # Add autocovariance matrices for negative lags, in order Gamma_0, Gamma_1, ..., Gamma_p, Gamma_{-p}, ..., Gamma_{-1}. autocov_mats[j] gives the correct matrix for j = -p, ..., p.
    autocov_mats = np.vstack([autocov_mats, autocov_mats[1:][::-1].transpose(0, 2, 1)])
    # Add identity matrix to ns_mats (stage 0 weights) and transpose so that each row of each matrix gives the weights for a given node 
    ns_weights = np.vstack([np.eye(d).reshape(1, d, d), ns_mats.transpose(0, 2, 1)])

    # Construct the GNAR autocovariance matrices for each node, lag and stage of neighbour dependence
    gnar_autocovs = np.zeros([d, 2 * p + 1, r + 1, r + 1])
    for i in range(d):
        gnar_autocovs[i] = ns_weights[:, i] @ autocov_mats @ ns_weights[:, i].T
    return gnar_autocovs

def structure_covariance_mats(autocov_mats, ns_mats, p, s):
    """
    Given the network autocovariance matrices for tau = 0, 1, ..., p, construct the GNAR autocovariance matrices for the Yule-Walker equations.
    
    Params:
        autocov_mats: np.ndarray of shape (p + 1, d, d). Covariance matrices for tau = 0, 1, ..., p.
        ns_mats: np.ndarray of shape (max(s), d, d). Neighbour sum matrices for r = 1, ..., max(s).
        p: int. Number of lags.
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)
    """
    r, d, _ = ns_mats.shape
    # Compute the network autocovariance matrices for each node, lag and stage of neighbour dependence
    gnar_autocovs = network_covariance_mats(autocov_mats, ns_mats, p)

    # Construct the Yule-Walker equations for each node separately - these will then be adapted based on the model type
    # These are already stacked into single matrices for each node
    gamma_G = np.zeros([d, p + np.sum(s)])
    Gamma_G = np.zeros([d, p + np.sum(s), p + np.sum(s)])
    s_tau = 0
    for tau in range(1, p + 1):
        # The sizes of each individual gamma_G and Gamma_G matrix are determined by the stage of neighbour dependence for the current lags
        gamma_G[:, s_tau : s_tau + 1 + s[tau - 1]] = gnar_autocovs[:, tau, 0, : 1 + s[tau - 1]]
        s_j = 0
        for j in range(1, p + 1):
            Gamma_G[:, s_tau : s_tau + 1 + s[tau - 1], s_j : s_j + 1 + s[j - 1]] = gnar_autocovs[:, j - tau, : 1 + s[tau - 1], : 1 + s[j - 1]]
            s_j += 1 + s[j - 1]
        s_tau += 1 + s[tau - 1]

    return gamma_G, Gamma_G

def gnar_yw(autocov_mats, ns_mats, p, s, model_type):
    """
    Fit a GNAR model using the Yule-Walker equations.
    
    Params:
        autocov_mats: np.ndarray of shape (p + 1, d, d). Covariance matrices for tau = 0, 1, ..., p.
        ns_mats: np.ndarray of shape (max(s), d, d). Neighbour sum matrices for r = 1, ..., max(s).
        p: int. Number of lags.
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)
        model_type: str. Type of GNAR model to fit. One of "global", "standard" or "local".
    """
    # Data shape
    r, d, _ = ns_mats.shape
    # Compute the covariance matrices for tau = 0, 1, ..., p
    gamma_G, Gamma_G = structure_covariance_mats(autocov_mats, ns_mats, p, s)

    # Solve the Yule-Walker equations for the given model type
    if model_type == "global":
        # Sum the gamma_G vectors and the Gamma_G matrices across nodes
        y = np.sum(gamma_G, axis=0)
        X = np.sum(Gamma_G, axis=0)
        # Set coefficients to zero if the corresponding column of the covariance matrix is all zeros
        valid_cols = np.any(X != 0, axis=0)
        coeffs = np.zeros(len(X))
        coeffs[valid_cols] = solve(X[valid_cols, :][:, valid_cols], y[valid_cols])
        # Remap the coefficients to the original shape
        coeffs_mat = np.tile(coeffs, (d, 1))

    elif model_type == "standard":
        X, y = format_standard(Gamma_G, gamma_G, d, p, s)
        # Set coefficients to zero if the corresponding column of the covariance matrix is all zeros
        valid_cols = np.any(X != 0, axis=0)
        # Use scipy's sparse matrix solver for efficiency since X can contain many zeros
        X_sparse = csr_matrix(X[valid_cols, :][:, valid_cols])
        coeffs = np.zeros(len(X))
        coeffs[valid_cols] = spsolve(X_sparse, y[valid_cols])
        # Remap the coefficients into a matrix
        coeffs_mat = np.zeros([d, p + np.sum(s)])
        s_j = 0
        for i in range(p):
            coeffs_mat[:, i + s_j: i + 1 + s_j] = coeffs[d * i + s_j: d * (i + 1) + s_j].reshape(-1, 1)
            coeffs_mat[:, i + 1 + s_j :  i + 1 + s_j + s[i]] = np.tile(coeffs[d * (i + 1) + s_j : d * (i + 1) + s_j + s[i]], (d, 1))
            s_j += s[i]

    elif model_type == "local":
        coeffs_mat = np.zeros((d, p + np.sum(s)))
        # Solve the Yule-Walker equations for each node separately
        for i in range(d):
            y = gamma_G[i]
            X = Gamma_G[i]
            # Set coefficients to zero if the corresponding column of the covariance matrix is all zeros
            valid_cols = np.any(X != 0, axis=0)
            coeffs_mat[i, valid_cols] = solve(X[valid_cols, :][:, valid_cols], y[valid_cols])
    else:
        raise ValueError("Invalid model type. Expected one of 'global', 'standard' or 'local'.")
    
    # Reorder the coefficients matrix to match the order stored in the GNAR class
    coeffs_mat = rearrange(coeffs_mat, p, s)
    return coeffs_mat


def format_standard(Gamma_G, gamma_G, d, p, s):
    """
    Formats each section of the Gamma_G matrix and gamma_G vector for estimation in the standard setting.
    Essentially we are combining d Yule-Walker equations with y_i of shape (d + sum(s)) and X_i of shape (d + sum(s), d + sum(s)) into a single equation with y of shape (d + sum(s)) and X of shape (d + sum(s), d + sum(s)).
    """
    X = np.zeros([p * d + np.sum(s), p * d + np.sum(s)])
    y = np.zeros(p * d + np.sum(s))
    # Keep track of the current position in the y vector and X matrix
    s_tau = 0
    s_tau_X = 0
    for tau in range(1, p + 1):
        gamma_G_tau = gamma_G[:, s_tau : s_tau + 1 + s[tau - 1]]
        y_tau = np.zeros(d + s[tau - 1])
        # The alpha coefficients are unconstrained across nodes
        y_tau[:d] = gamma_G_tau[:, 0]
        # The betas are constrained to be the same
        y_tau[d:] = np.sum(gamma_G_tau[:, 1:], axis=0)
        # Add this to the y vector
        y[s_tau_X : s_tau_X + d + s[tau - 1]] = y_tau
        # Keep track of the current position in the X matrix
        s_j = 0
        s_j_X = 0
        for j in range(1, p + 1):
            Gamma_G_tau_j = Gamma_G[:, s_tau : s_tau + 1 + s[tau - 1], s_j : s_j + 1 + s[j - 1]]
            X_tau_j = np.zeros([d + s[tau - 1], d + s[tau - 1]])
            # The alpha coefficients are unconstrained across nodes, so the corresponding data is placed diagonally in the matrix
            X_tau_j[:d, :d] = np.diag(Gamma_G_tau_j[:, 0, 0])
            # The betas are constrained to be the same, so the corresponding data is placed in the remaining columns/rows and in some cases summed
            X_tau_j[:d, d:] = Gamma_G_tau_j[:, 0, 1:]
            X_tau_j[d:, :d] = Gamma_G_tau_j[:, 1:, 0].T
            X_tau_j[d:, d:] = np.sum(Gamma_G_tau_j[:, 1:, 1:], axis=0)
            # Add this to the X matrix
            X[s_tau_X : s_tau_X + d + s[tau - 1], s_j_X : s_j_X + d + s[j - 1]] = X_tau_j
            # Update the current position in the X matrix
            s_j += 1 + s[j - 1]
            s_j_X += d + s[j - 1]
        # Update the current position in the y vector and X matrix
        s_tau += 1 + s[tau - 1]
        s_tau_X += d + s[tau - 1]
    return X, y


def rearrange(coeffs_mat, p, s):
    """
    Rearrange the coefficients matrix to match the order stored in the GNAR class.
    """
    coeffs_ordered = np.zeros_like(coeffs_mat)
    s_j = 0
    # Iterate through each lag. Store all alpha coefficients first, followed by all beta coefficients
    for j in range(p):
        coeffs_ordered[:, j] = coeffs_mat[:, j + s_j]
        coeffs_ordered[:, p + s_j : p + s_j + s[j]] = coeffs_mat[:, 1 + j + s_j : 1 + j + s_j + s[j]]
        s_j += s[j]
    return coeffs_ordered


def estimate_res_mat(data, coeffs_mat, p, s, n):
    """
    Estimate the matrix of residuals for the GNAR model fit via the Yule-Walker equations.
    """
    X, y = format_X_y(data, p, s)
    # Compute the covariance matrix of the residuals
    res = np.sum(X * coeffs_mat, axis=2) - y
    sigma_2 = res.T @ res / (n - p)
    return sigma_2