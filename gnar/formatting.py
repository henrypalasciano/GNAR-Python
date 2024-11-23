import numpy as np
import pandas as pd

def parameter_dataframe(p, s_vec, ts=None, parameters=None):
    """
    Create a dataframe to store the parameters of the GNAR model.

    Parameters:
        p (int): The number of lags.
        s_vec (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag.
        ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
        parameters (np.ndarray): The parameters of the GNAR model. Shape (n, p + sum(s_vec)).

    Returns:
        pd.DataFrame. Dataframe to store the pramaters of the GNAR model.
    """
    # Index of the dataframe
    index = ["mean"] + ["a_" + str(i) for i in range(1, p + 1)]
    for i in range(1, p + 1):
        index += ["b_" + str(i) + "," + str(j) for j in range(1, s_vec[i - 1] + 1)]
    # Columns of the dataframe
    if ts is not None:
        if isinstance(ts, np.ndarray):
            columns = np.arange(ts.shape[1])
        else:
            columns = ts.columns
        return pd.DataFrame(index=index, columns=columns, dtype=float)

    elif parameters is not None:
        columns = np.arange(parameters.shape[1])
        return pd.DataFrame(parameters, index=index, columns=columns, dtype=float)

    else:
        raise ValueError("Either the input time series data or the coefficients must be provided.")
        
    