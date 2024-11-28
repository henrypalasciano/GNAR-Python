import numpy as np
import pandas as pd

def parameter_df(p, d, s=None, ts=None, parameters=None):
    """
    Create a dataframe to store the parameters of the GNAR model.

    Parameters:
        p (int): The number of lags.
        d (int): The number of time series.
        s (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag. If not provided, the model is assumed to be a VAR.
        ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, d) where m is the number of observations and d is the number of nodes.
        parameters (np.ndarray): The parameters of the GNAR model. Shape (d, p + sum(s)).

    Returns:
        pd.DataFrame. Dataframe to store the pramaters of the GNAR model.
    """
    # Index of the dataframe
    if s is None:
        index = ["mean"] + [f"a_{i},{j}" for i in range(1, p + 1) for j in range(d)]
    else:
        index = ["mean"] + [f"a_{i}" for i in range(1, p + 1)]
        for i in range(1, p + 1):
            index += [f"b_{i},{j}" for j in range(1, s[i - 1] + 1)]
    # Columns of the dataframe
    if ts is not None:
        if isinstance(ts, np.ndarray):
            columns = np.arange(d)
        else:
            columns = ts.columns
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    elif parameters is not None:
        columns = np.arange(d)
        return pd.DataFrame(parameters, index=index, columns=columns, dtype=float)

    else:
        raise ValueError("Either the input time series data or the coefficients are required.") 

def cov_df(sigma_2, names):
    """
    Convert sigma_2 to a dataframe.

    Parameters:
        sigma_2 (int, float, np.ndarray or pd.DataFrame): The variance or covariance matrix.
        names (list): The names of the nodes.
    
    Returns:
        pd.DataFrame. The covariance matrix as a dataframe.
    """
    if sigma_2 is None:
        return pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    elif isinstance(sigma_2, (int, float)):
        return pd.DataFrame(sigma_2 * np.eye(len(names)), index=names, columns=names)
    elif isinstance(sigma_2, np.ndarray):
        if sigma_2.ndim == 1 or sigma_2.shape[0] == 1:
            return pd.DataFrame(np.diag(sigma_2.flatten()), index=names, columns=names)
        return pd.DataFrame(sigma_2, index=names, columns=names)
    return sigma_2