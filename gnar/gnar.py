import numpy as np
import pandas as pd

from model_fitting import format_X_y, gnar_lr
from forecasting import format_X, update_X
from neighbour_sets import neighbour_set_tensor
from fomulate import coefficient_dataframe

class GNAR:
    def __init__(self, ts, A, p, s_vec, intercept=True, model_type="standard"):
        """
        Initialize the GNAR model.

        Parameters:
            ts (ndarray): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            A (ndarray): The adjacency matrix. Shape (n, n).
            p (int): The order of the GNAR model.
            s_vec (ndarray): The maximum stage of neighbour dependence for each lag. Shape (p,)
        """
        self.ts = ts
        self.A = A
        # Compute the tensor of matrices conrresponding to the neighbour sets of each node
        self.A_tensor = neighbour_set_tensor(A, max(s_vec))
        self.p = p
        self.s_vec = s_vec
        self.intercept = intercept
        self.model_type = model_type
        self.fit()

    def fit(self):
        """
        Fit the GNAR model to the given data.

        Parameters:
            p (int): The number of lags.
            s (ndarray): An array containing the maximum stage of neighbour dependence for each lag.
        """
        m, n = np.shape(self.ts)
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + max(self.s_vec)])
        data[:, :, 0] = self.ts
        data[:, :, 1:] = np.transpose(self.ts @ self.A_tensor, (1, 2, 0))
        X,y = format_X_y(data, self.p, self.s_vec)
        # Fit the model for each node using least squares regression
        self.coefficients = gnar_lr(X, y, self.p, self.s_vec, self.intercept, self.model_type)

    def predict(self, ts, n_steps=1):
        """
        Predict future values using the GNAR model.

        Parameters:
            ts (ndarray): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            n_steps (int): The number of future steps to predict.

        Returns:
            predictions (ndarray): The predicted values. Shape (m - 1 + p, n, n_steps)
        """
        # Data shapes
        m, n = np.shape(ts)
        r = np.max(self.s_vec)
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + r])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self.A_tensor, (1, 2, 0))
        # Format the data to make predictions. The laggad values array contains lags for the time series and neighbour sums up to the maximum neighbour stage of dependence. This is an array of shape (m - p + 1, n, p * (1 + max(s))), which is required for updating X.
        X, lagged_vals = format_X(data, self.p, self.s_vec, self.intercept)
        # Initialise the array to store the predictions, which is an array of shape (m - p + 1, n, n_steps)
        predictions = np.zeros([m - self.p + 1, n, n_steps])
        # Compute the one-step ahead predictions
        predictions[:, :, 0] = np.sum(X * self.coefficients, axis=2)
        for i in range(1, n_steps):
            # Update the lagged values and design matrix using the predicted values
            lagged_vals = np.dstack([np.transpose(predictions[:, :, i-1] @ self.A_tensor, (1, 2, 0)), lagged_vals[:, :, :-r]])
            # Update the design matrix
            X = update_X(X, predictions[:, :, i-1], lagged_vals, self.p, self.s_vec, self.intercept)
            # Compute the (i + 1) - step ahead predictions
            predictions[:, :, i] = np.sum(X * self.coefficients, axis=2)
        return predictions


class GNAR_df:
    def __init__(self, ts, A, p, s_vec, intercept=False, model_type="standard"):
        """
        Initialize the GNAR model, where inputs and outputs are pandas DataFrames.

        Parameters:
            ts (pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            A (ndarray): The adjacency matrix. Shape (n, n).
            p (int): The order of the GNAR model.
            s_vec (ndarray): The maximum stage of neighbour dependence for each lag. Shape (p,)
            intercept (bool): Whether to include an intercept in the model.
        """
        self.ts = ts
        self.G = GNAR(ts.to_numpy(), A, p, s_vec, intercept, model_type)
        self.coefficients = coefficient_dataframe(ts, p, s_vec, intercept)
        self.coefficients.iloc[:, :] = self.G.coefficients        

    def predict(self, ts, n_steps=1):
        preds = self.G.predict(ts.to_numpy(), n_steps)
        index = ts.index[self.G.p - 1:]

        ts_names = ts.columns
        steps = range(1, n_steps + 1)
        # Create a multi-index for the columns
        columns = pd.MultiIndex.from_product([ts_names, steps], names=['Time Series', 'Steps Ahead'])
        # Create a DataFrame with the multi-index columns
        predictions = pd.DataFrame(index=index, columns=columns, dtype=float)
        for i in range(n_steps):
            predictions.loc[:, (ts_names, i + 1)] = preds[:, :, i]
        return predictions
