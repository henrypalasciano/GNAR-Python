import numpy as np
import pandas as pd

from model_fitting import format_X_y, gnar_lr
from forecasting import format_X, update_X
from neighbour_sets import neighbour_set_tensor
from formatting import coefficient_dataframe

class GNAR:
    def __init__(self, A, p, s_vec, ts=None, remove_mean=True, coefficients=None, mu=None, model_type="standard"):
        """
        Initialize the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            A (np.ndarray): The adjacency matrix. Shape (n, n).
            p (int): The order of the GNAR model.
            s_vec (np.ndarray): The maximum stage of neighbour dependence for each lag. Shape (p,)
        """
        self._A = A
        # Compute the tensor of matrices conrresponding to the neighbour sets of each node
        self._A_tensor = neighbour_set_tensor(A, max(s_vec))
        self._p = p
        self._s_vec = s_vec
        self._model_type = model_type
        self._model_setup(ts, remove_mean, coefficients, mu)

    def _model_setup(self, ts=None, remove_mean=True, coefficients=None, mu=None):
        """
        Setup the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            coefficients (np.ndarray): The coefficients of the GNAR model. Shape (n, p + sum(s_vec)).
            mu (np.ndarray): The mean of the input time series data. Shape (1, n).
        """
        if ts is not None:
            self._coefficients = coefficient_dataframe(self._p, self._s_vec, ts=ts)
            self._mu = pd.DataFrame(columns=self._coefficients.columns, index=["mean"], dtype=float)
            if isinstance(ts, np.ndarray):
                self.fit(ts, remove_mean)
            else:
                self.fit(ts.to_numpy(), remove_mean)
        elif coefficients is not None:
            if isinstance(coefficients, pd.DataFrame):
                self._coefficients = coefficients
            else:
                self._coefficients = coefficient_dataframe(self._p, self._s_vec, coefficients=coefficients)
            if mu is not None:
                if isinstance(mu, np.ndarray):
                    self._mu = pd.DataFrame(mu.reshape(1, coefficients.shape[1]), columns=self._coefficients.columns, index=["mean"], dtype=float)
                else:
                    self._mu = mu
            else:
                self._mu = pd.DataFrame(np.zeros((1, coefficients.shape[1])), columns=self._coefficients.columns, index=["mean"], dtype=float)
        else:
            raise ValueError("Either the input time series data or the coefficients must be provided.")

    def fit(self, ts, remove_mean=True):
        """
        Fit the GNAR model to the given data.

        Parameters:
            p (int): The number of lags.
            s (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag.
        """
        m, n = np.shape(ts)
        # Compute the mean if necessary and save this to a DataFrame
        if remove_mean:
            mu = np.mean(ts, axis=0, keepdims=True)
        else:
            mu = np.zeros((1, n))
        ts = ts - mu
        self._mu.iloc[:, :] = mu

        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + max(self._s_vec)])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._A_tensor, (1, 2, 0))
        X,y = format_X_y(data, self._p, self._s_vec)
        # Fit the model for each node using least squares regression and save the coefficients to a DataFrame
        self._coefficients.iloc[:, :] = gnar_lr(X, y, self._p, self._s_vec, self._model_type).T

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
        if n != self._coefficients.shape[1]:
            raise ValueError("The number of nodes in the input time series does not match the number of nodes in the model.")
        r = np.max(self._s_vec)

        # Convert the predictions to a pandas DataFrame if the input time series is a DataFrame
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            ts_names = ts.columns
            columns = pd.MultiIndex.from_product([ts_names, range(1, n_steps + 1)], names=['Time Series', 'Steps Ahead'])
            predictions_df = pd.DataFrame(index=ts.index[self._p - 1:], columns=columns, dtype=float)
            ts = ts.to_numpy()
        
        mu = self._mu.to_numpy()
        coefficients = self._coefficients.to_numpy().T
        
        ts -= mu
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + r])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._A_tensor, (1, 2, 0))
        # Format the data to make predictions. The laggad values array contains lags for the time series and neighbour sums up to the maximum neighbour stage of dependence. This is an array of shape (m - p + 1, n, p * (1 + max(s))), which is required for updating X.
        X, lagged_vals = format_X(data, self._p, self._s_vec)
        
        # Initialise the array to store the predictions, which is an array of shape (m - p + 1, n, n_steps)
        predictions = np.zeros([m - self._p + 1, n, n_steps])
        # Compute the one-step ahead predictions
        predictions[:, :, 0] = np.sum(X * coefficients, axis=2)
        if is_df:
            predictions_df.loc[:, (ts_names, 1)] = predictions[:, :, 0] + mu
        for i in range(1, n_steps):
            # Update the lagged values and design matrix using the predicted values
            lagged_vals = np.dstack([np.transpose(predictions[:, :, i-1] @ self._A_tensor, (1, 2, 0)), lagged_vals[:, :, :-r]])
            # Update the design matrix
            X = update_X(X, predictions[:, :, i-1], lagged_vals, self._p, self._s_vec)
            # Compute the (i + 1) - step ahead predictions
            predictions[:, :, i] = np.sum(X * coefficients, axis=2)
            if is_df:
                predictions_df.loc[:, (ts_names, i + 1)] = predictions[:, :, i] + mu
        
        if is_df:
            return predictions_df
        return predictions + mu.reshape(1, n, 1)

    def __str__(self):
        model_info = f"{self._model_type.capitalize()} GNAR({self._p}, {self._s_vec}) Model\n"
        parameter_info = f"Parameters:\n{pd.concat([self._mu, self._coefficients], axis=0)}\n"
        return model_info + parameter_info