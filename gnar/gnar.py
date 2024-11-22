import numpy as np
import pandas as pd

from model_fitting import format_X_y, gnar_lr
from forecasting import format_X, update_X
from neighbour_sets import neighbour_set_tensor
from formatting import coefficient_dataframe

class GNAR:
    def __init__(self, ts, A, p, s_vec, remove_mean=True, model_type="standard"):
        """
        Initialize the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            A (np.ndarray): The adjacency matrix. Shape (n, n).
            p (int): The order of the GNAR model.
            s_vec (np.ndarray): The maximum stage of neighbour dependence for each lag. Shape (p,)
        """
        if isinstance(ts, np.ndarray):
            self.ts = ts.copy()
        else:
            self.ts = ts.to_numpy().copy()
        self.A = A
        # Compute the tensor of matrices conrresponding to the neighbour sets of each node
        self.A_tensor = neighbour_set_tensor(A, max(s_vec))
        self.p = p
        self.s_vec = s_vec
        if remove_mean:
            self.mu = np.mean(self.ts, axis=0, keepdims=True)
            self.ts -= self.mu
        else:
            self.mu = np.zeros((1, self.ts.shape[1]))
        self.model_type = model_type
        # Model coefficients are stored in a dataframe, where the index are the nodes and the columns are the coefficients
        self.coefficients = coefficient_dataframe(ts, p, s_vec)
        self.fit()

    def fit(self):
        """
        Fit the GNAR model to the given data.

        Parameters:
            p (int): The number of lags.
            s (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag.
        """
        m, n = np.shape(self.ts)
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + max(self.s_vec)])
        data[:, :, 0] = self.ts
        data[:, :, 1:] = np.transpose(self.ts @ self.A_tensor, (1, 2, 0))
        X,y = format_X_y(data, self.p, self.s_vec)
        # Fit the model for each node using least squares regression
        self.coefficients.iloc[:, :] = gnar_lr(X, y, self.p, self.s_vec, self.model_type)

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
        m, n = ts.shape
        if n != self.coefficients.shape[0]:
            raise ValueError("The number of nodes in the input time series does not match the number of nodes in the model.")
        r = np.max(self.s_vec)

        # Convert the predictions to a pandas DataFrame if the input time series is a DataFrame
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            ts_names = ts.columns
            columns = pd.MultiIndex.from_product([ts_names, range(1, n_steps + 1)], names=['Time Series', 'Steps Ahead'])
            predictions_df = pd.DataFrame(index=ts.index[self.p - 1:], columns=columns, dtype=float)
            ts = ts.to_numpy()
        
        ts -= self.mu
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + r])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self.A_tensor, (1, 2, 0))
        # Format the data to make predictions. The laggad values array contains lags for the time series and neighbour sums up to the maximum neighbour stage of dependence. This is an array of shape (m - p + 1, n, p * (1 + max(s))), which is required for updating X.
        X, lagged_vals = format_X(data, self.p, self.s_vec)
        coefficients = self.coefficients.to_numpy()
        
        # Initialise the array to store the predictions, which is an array of shape (m - p + 1, n, n_steps)
        predictions = np.zeros([m - self.p + 1, n, n_steps])
        # Compute the one-step ahead predictions
        predictions[:, :, 0] = np.sum(X * coefficients, axis=2)
        if is_df:
            predictions_df.loc[:, (ts_names, 1)] = predictions[:, :, 0] + self.mu
        for i in range(1, n_steps):
            # Update the lagged values and design matrix using the predicted values
            lagged_vals = np.dstack([np.transpose(predictions[:, :, i-1] @ self.A_tensor, (1, 2, 0)), lagged_vals[:, :, :-r]])
            # Update the design matrix
            X = update_X(X, predictions[:, :, i-1], lagged_vals, self.p, self.s_vec)
            # Compute the (i + 1) - step ahead predictions
            predictions[:, :, i] = np.sum(X * coefficients, axis=2)
            if is_df:
                predictions_df.loc[:, (ts_names, i + 1)] = predictions[:, :, i] + self.mu
        
        if is_df:
            return predictions_df
        return predictions + self.mu.reshape(1, n, 1)