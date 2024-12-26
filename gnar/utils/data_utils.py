import numpy as np
import pandas as pd

def set_mean(mean, d):
    if isinstance(mean, (float, int)):
        return np.full((1, d), mean)
    elif isinstance(mean, np.ndarray):
        return mean.reshape(1, d)
    elif isinstance(mean, pd.DataFrame):
        return mean.to_numpy().reshape(1, d)
    else:
        raise ValueError("Mean must be a float, int, NumPy array or Pandas DataFrame.")

def set_cov(sigma_2):
    if isinstance(sigma_2, (float, int, np.ndarray)):
        return sigma_2
    elif isinstance(sigma_2, pd.DataFrame):
        return sigma_2.to_numpy()
    else:
        raise ValueError("Noise covariance matrix must be a float, int, NumPy array or Pandas DataFrame.")

def cov_mat(sigma_2, d):
    if isinstance(sigma_2, (int, float)):
        return sigma_2 * np.eye(d)
    elif isinstance(sigma_2, np.ndarray) and (sigma_2.ndim == 1 or sigma_2.shape[0] == 1):
        return np.diag(sigma_2.flatten())
    return sigma_2

def check_gnar_coeffs(coeffs, d, p, s, model_type):
    k, d = np.shape(coeffs)
    if d != A.shape[0]:
        raise ValueError("The number of nodes in the adjacency matrix does not match the number of nodes in the coefficients matrix.")
    if k != p + np.sum(s):
        raise ValueError("The number of coefficients does not match the number of parameters.")
    unique_params = np.apply_along_axis(lambda row: len(np.unique(row)), axis=1, arr=coeffs)
    if model_type == "global" and np.any(unique_params != 1):
        raise ValueError("The model type is global, but the coefficients are not the same for all nodes.")
    elif model_type == "standard" and np.any(unique_params[d * p:] != 1):
        raise ValueError("The model type is standard, but the beta coefficients are not the same for all nodes.")
    return None