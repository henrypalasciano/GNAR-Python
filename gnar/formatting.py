import numpy as np
import pandas as pd

def coefficient_dataframe(ts, p, s_vec):
    """
    Create a dataframe to store the coefficients of the GNAR model.

    Parameters:
        ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
        p (int): The number of lags.
        s_vec (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag.

    Returns:
        pd.DataFrame. Dataframe to store the coefficients of the GNAR model.
    """
    # Index of the dataframe
    if isinstance(ts, np.ndarray):
        index = np.arange(ts.shape[1])
    else:
        index = ts.columns
        
    # Columns of the dataframe
    columns = ["a_" + str(i) for i in range(1, p + 1)]
    for i in range(1, p + 1):
        columns += ["b_" + str(i) + "," + str(j) for j in range(1, s_vec[i - 1] + 1)]
    return pd.DataFrame(index=index, columns=columns)