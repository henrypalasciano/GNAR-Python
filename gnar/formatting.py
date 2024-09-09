import pandas as pd

def coefficient_dataframe(ts, p, s_vec, intercept):
    """
    Create a dataframe to store the coefficients of the GNAR model.

    Parameters:
        ts (pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
        p (int): The number of lags.
        s_vec (ndarray): An array containing the maximum stage of neighbour dependence for each lag.
        intercept (bool): Whether to include an intercept in the model.

    Returns:
        pd.DataFrame. Dataframe to store the coefficients of the GNAR model.
    """
    # Index of the dataframe
    index = ts.columns
    # Columns of the dataframe
    if intercept:
        columns = ["a_0"]
    else:
        columns = []
    columns += ["a_" + str(i) for i in range(1, p + 1)]
    for i in range(1, p + 1):
        columns += ["b_" + str(i) + "," + str(j) for j in range(1, s_vec[i - 1] + 1)]
    return pd.DataFrame(index=index, columns=columns)