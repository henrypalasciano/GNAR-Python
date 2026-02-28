import numpy as np
import pandas as pd

def gnar_checks(A: np.ndarray, p: int, s: np.ndarray, model_type: str, net_type: str = "unweighted") -> None:
    if not isinstance(A, np.ndarray) or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be a square NumPy array.")
    if np.any(A < 0):
        raise ValueError("Adjacency matrix A must have non-negative weights.")
    valid_net_types = {"unweighted", "weighted", "distance"}
    if net_type not in valid_net_types:
        raise ValueError(f"Invalid net_type. Expected one of {valid_net_types}, got '{net_type}'.")
    if net_type == "unweighted" and not np.all((A == 0) | (A == 1)):
        raise ValueError("Adjacency matrix A must be binary (0 or 1) for unweighted networks.")
    if p < 1:
        raise ValueError("The number of lags p must be at least 1.")
    if not isinstance(s, np.ndarray) or len(s) != p:
        raise ValueError("Array s must be a NumPy array with length equal to p (number of lags).")
    if np.any(s < 0):
        raise ValueError("Values in s must be non-negative integers.")
    valid_model_types = {"global", "standard", "local"}
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type. Expected one of {valid_model_types}, got '{model_type}'.")
    return None

def set_mean(mean: float | int | np.ndarray | pd.DataFrame, d: int) -> np.ndarray:
    if isinstance(mean, (float, int)):
        return np.full((1, d), mean)
    elif isinstance(mean, np.ndarray):
        return mean.reshape(1, d)
    elif isinstance(mean, pd.DataFrame):
        return mean.to_numpy().reshape(1, d)
    else:
        raise ValueError("Mean must be a float, int, NumPy array or Pandas DataFrame.")

def set_cov(sigma_2: float | int | np.ndarray | pd.DataFrame) -> float | int | np.ndarray:
    if isinstance(sigma_2, (float, int, np.ndarray)):
        return sigma_2
    elif isinstance(sigma_2, pd.DataFrame):
        return sigma_2.to_numpy()
    else:
        raise ValueError("Noise covariance matrix must be a float, int, NumPy array or Pandas DataFrame.")

def cov_mat(sigma_2: float | int | np.ndarray, d: int) -> np.ndarray:
    if isinstance(sigma_2, (int, float)):
        return sigma_2 * np.eye(d)
    elif isinstance(sigma_2, np.ndarray) and (sigma_2.ndim == 1 or sigma_2.shape[0] == 1):
        return np.diag(sigma_2.flatten())
    return sigma_2

def check_gnar_coeffs(coeffs: np.ndarray, d: int, p: int, s: np.ndarray, model_type: str) -> None:
    k, q = np.shape(coeffs)
    if d != q:
        raise ValueError("The number of nodes in the adjacency matrix does not match the number of nodes in the coefficients matrix.")
    if k != p + np.sum(s):
        raise ValueError("The number of coefficients does not match the number of parameters.")
    unique_params = np.apply_along_axis(lambda row: len(np.unique(row)), axis=1, arr=coeffs)
    if model_type == "global" and np.any(unique_params != 1):
        raise ValueError("The model type is global, but the coefficients are not the same for all nodes.")
    elif model_type == "standard" and np.any(unique_params[p:] != 1):
        raise ValueError("The model type is standard, but the beta coefficients are not the same for all nodes.")
    return None

def param_reorder(coeffs: np.ndarray, p: int, s: np.ndarray) -> np.ndarray:
    """
    Reorder the coefficients to match the structure of the weight matrices.
    """
    order = []
    s_tau = 0
    for i in range(p):
        order.append(i)
        order += list(range(p + s_tau, p + s_tau + s[i]))
        s_tau += s[i]
    return coeffs[order]
