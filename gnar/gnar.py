import numpy as np
import pandas as pd
import networkx as nx

from model_fitting import format_X_y, gnar_lr
from forecasting import format_X, update_X
from simulating import shift_X
from neighbour_sets import neighbour_set_tensor
from formatting import parameter_dataframe

class GNAR:
    def __init__(self, A, p, s_vec, ts=None, remove_mean=True, parameters=None, sigma_2=None, model_type="standard"):
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
        self._model_setup(ts, remove_mean, parameters, sigma_2)
        self._to_networkx()

    def _model_setup(self, ts=None, remove_mean=True, parameters=None, sigma_2=None):
        """
        Setup the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            parameters (np.ndarray): The parameters of the GNAR model. Shape (n, p + sum(s_vec)).
        """
        if ts is not None:
            self._parameters = parameter_dataframe(self._p, self._s_vec, ts=ts)
            if isinstance(ts, np.ndarray):
                self.fit(ts, remove_mean)
            else:
                self.fit(ts.to_numpy(), remove_mean)
        elif parameters is not None:
            if isinstance(parameters, pd.DataFrame):
                self._parameters = parameters
            else:
                self._parameters = parameter_dataframe(self._p, self._s_vec, parameters=parameters)
            if sigma_2 is None:
                self._sigma_2 = 1
            elif isinstance(sigma_2, (float, int, np.ndarray)):
                self._sigma_2 = sigma_2
            else:
                self._sigma_2 = sigma_2.to_numpy()
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
        self._parameters.loc["mean", :] = mu

        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + max(self._s_vec)])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._A_tensor, (1, 2, 0))
        X,y = format_X_y(data, self._p, self._s_vec)
        # Fit the model for each node using least squares regression and save the coefficients to a DataFrame
        coefficients =  gnar_lr(X, y, self._p, self._s_vec, self._model_type)
        res = np.sum(X * coefficients, axis=2) - y
        self._sigma_2 = res.T @ res / (m - self._p)
        self._parameters.iloc[1:, :] = coefficients.T

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
        if n != self._parameters.shape[1]:
            raise ValueError("The number of nodes in the input time series does not match the number of nodes in the model.")
        r = np.max(self._s_vec)

        # Convert the predictions to a pandas DataFrame if the input time series is a DataFrame
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            ts_names = ts.columns
            columns = pd.MultiIndex.from_product([ts_names, range(1, n_steps + 1)], names=['Time Series', 'Steps Ahead'])
            predictions_df = pd.DataFrame(index=ts.index[self._p - 1:], columns=columns, dtype=float)
            ts = ts.to_numpy()
        
        mu = self._parameters.to_numpy()[0]
        coefficients = self._parameters.to_numpy()[1:].T
        
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

    def simulate(self, m, sigma_2=None, burn_in=50):
        """
        Simulate data from the GNAR model.

        Parameters:
            m (int): The number of time steps to simulate.
            sigma_2 (float or np.ndarray): The variance of the noise. If a float, the same variance is used for all nodes. If an array, the variances are used for each node.
            burn_in (int): The number of burn-in steps to discard.
        
        Returns:
            ts_sim (np.ndarray): The simulated time series data. Shape (m, n)
        """
        # Number of nodes and coefficients
        n = self._parameters.shape[1]
        r = np.max(self._s_vec)
        coefficients = self._parameters.iloc[1:, :].to_numpy().T

        # Generate the noise
        if sigma_2 is None:
            sigma_2 = self._sigma_2
        if isinstance(sigma_2, (float, int)):
            e_t = np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(burn_in + m, n))
        elif sigma_2.ndim == 1:
            e_t = np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(burn_in + m, n))
        elif sigma_2.shape[0] == 1:
            e_t = np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(burn_in + m, n))
        else:
            e_t = np.random.multivariate_normal(mean=np.zeros(n), cov=sigma_2, size=burn_in + m)
        
        # Initialise the array to store the simulated time series data
        ts_sim = np.zeros([burn_in + m, n])
        ns = np.zeros([n, self._p * r])
        # Initialise the design matrix, which is an array of shape (n, p + sum(s))
        X = np.zeros([n, self._p + np.sum(self._s_vec)])
        for t in range(self._p, burn_in + m):
            # Compute the simulated observation
            sim = np.sum(X * coefficients, axis=1) + e_t[t]
            ts_sim[t] = sim
            # Update neighbour sums array
            if self._p == 1:
                ns = (sim @ self._A_tensor).T
            else:
                ns = np.hstack([(sim @ self._A_tensor).T, ns[:, :-r]])
            # Shift the design matrix with the new simulated values and neighbour sums
            X = shift_X(X, sim, ns, self._p, self._s_vec)

        # Return the simulated time series data
        return ts_sim[burn_in:]

    def _to_networkx(self):
        """
        Convert the adjacency matrix to a NetworkX graph, which is stored in the nx_graph attribute.
        """
        self.nx_graph = nx.from_numpy_array(self._A)
        self.nx_graph = nx.relabel_nodes(self.nx_graph, dict(enumerate(self._parameters.columns)))

    def draw(self):
        """
        Draw the graph using NetworkX.
        """
        nx.draw(self.nx_graph, with_labels=True)

    def __str__(self):
        model_info = f"{self._model_type.capitalize()} GNAR({self._p}, {self._s_vec}) Model\n"
        graph_info = f"{self.nx_graph}\n"
        parameter_info = f"Parameters:\n{self._parameters}\n"
        return model_info + graph_info + parameter_info