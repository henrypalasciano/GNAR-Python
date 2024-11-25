import numpy as np
import pandas as pd
import networkx as nx

from utils.model_fitting import format_X_y, gnar_lr
from utils.forecasting import format_X, update_X
from utils.simulating import shift_X
from utils.neighbour_sets import neighbour_set_mats
from utils.formatting import parameter_df, cov_df
from var import VAR

class GNAR:
    def __init__(self, A, p, s_vec, ts=None, remove_mean=True, parameters=None, sigma_2=None, model_type="standard"):
        """
        Initialise the GNAR model.

        Parameters:
            A (np.ndarray): The adjacency matrix of the graph.
            p (int): The number of lags.
            s_vec (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag.
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            remove_mean (bool): Whether to remove the mean from the data. Only used if ts is provided.
            parameters (np.ndarray or pd.Datafram): The parameters of the GNAR model, consisting of the means and coefficients. Shape (1 + p + sum(s_vec), n).
            sigma_2 (float, int, np.ndarray or pd.DataFrame): The variance or covariance of the noise. If a float, the same variance is used for all nodes. Only used if parameters is provided.
            model_type (str): The type of GNAR model. Either "global", "standard" or "local".
        """
        self._A = A
        # Compute the neighbour set matrices up to the maximum stage of neighbour dependence
        self._ns_mats = neighbour_set_mats(A, max(s_vec))
        self._p = p
        self._s_vec = s_vec
        self._m = None
        self._n = A.shape[0]
        self._model_type = model_type
        self._model_setup(ts, remove_mean, parameters, sigma_2)
        self._to_networkx()

    def _model_setup(self, ts=None, remove_mean=True, parameters=None, sigma_2=None):
        """
        Setup the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            remove_mean (bool): Whether to remove the mean from the data. Only used if ts is provided.
            parameters (np.ndarray or pd.DataFrame): The parameters of the GNAR model, consisting of the means and coefficients. Shape (1 + p + sum(s_vec), n).
            sigma_2 (float, int, np.ndarray or pd.DataFrame): The variance or covariance of the noise. If a float, the same variance is used for all nodes. Only used if parameters is provided.
        """
        # If a time series is provided, fit the model to the data, removing the mean if necessary
        if ts is not None:
            self._parameters = parameter_df(self._p, self._n, s_vec=self._s_vec, ts=ts)
            if isinstance(ts, np.ndarray):
                self.fit(ts, remove_mean)
            else:
                self.fit(ts.to_numpy(), remove_mean)
        # If the parameters are provided, set the parameters and covariance matrix
        elif parameters is not None:
            if isinstance(parameters, pd.DataFrame):
                self._parameters = parameters
            else:
                self._parameters = parameter_df(self._p, self._n, s_vec=self._s_vec, parameters=parameters)
            if sigma_2 is None:
                self._sigma_2 = 1
            elif isinstance(sigma_2, (float, int, np.ndarray)):
                self._sigma_2 = sigma_2
            else:
                self._sigma_2 = sigma_2.to_numpy()
            self._cov = cov_df(sigma_2, self._parameters.columns)
        else:
            raise ValueError("Either the input time series data or the parameters must be provided.")

    def fit(self, ts, remove_mean=True):
        """
        Fit the GNAR model to the time series data.

        Parameters:
            ts (ndarray): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            remove_mean (bool): Whether to remove the mean from the data.
        """
        self._m, self._n = np.shape(ts)
        # Compute the mean if necessary and save this to the parameter DataFrame
        if remove_mean:
            mu = np.mean(ts, axis=0, keepdims=True)
        else:
            mu = np.zeros((1, self._n))
        ts = ts - mu
        self._parameters.loc["mean", :] = mu

        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([self._m, self._n, 1 + max(self._s_vec)])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._ns_mats, (1, 2, 0))
        X,y = format_X_y(data, self._p, self._s_vec)
        # Fit the model using least squares linear regression and save the coefficients to a DataFrame
        coefficients =  gnar_lr(X, y, self._p, self._s_vec, self._model_type)
        self._parameters.iloc[1:, :] = coefficients.T
        
        # Compute the noise covariance matrix
        res = np.sum(X * coefficients, axis=2) - y
        self._sigma_2 = res.T @ res / (self._m - self._p)
        # Store the covariance matrix as a DataFrame for display purposes
        self._cov = pd.DataFrame(self._sigma_2, index=self._parameters.columns, columns=self._parameters.columns, dtype=float)

    def predict(self, ts, n_steps=1):
        """
        Forecast future values of an input time series using the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (m, n) where m is the number of observations and n is the number of nodes.
            n_steps (int): The number of steps ahead to forecast.

        Returns:
            predictions (np.ndarray or pd.DataFrame): The predicted values. Shape (m - 1 + p, n, n_steps)
        """
        # Data shapes
        m, n = ts.shape
        if n != self._n:
            raise ValueError("The number of nodes in the input time series does not match the number of nodes in the model.")
        r = np.max(self._s_vec)

        # Convert the predictions to a pandas DataFrame if the input time series is a DataFrame
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            ts_names = ts.columns
            columns = pd.MultiIndex.from_product([ts_names, range(1, n_steps + 1)], names=["Time Series", "Steps Ahead"])
            predictions_df = pd.DataFrame(index=ts.index[self._p - 1:], columns=columns, dtype=float)
            ts = ts.to_numpy()
        
        # Get the mean and coefficients
        mu = self._parameters.to_numpy()[0]
        coefficients = self._parameters.to_numpy()[1:].T
        
        ts = ts - mu
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (m, n, max(s) + 1)
        data = np.zeros([m, n, 1 + r])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._ns_mats, (1, 2, 0))
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
            lagged_vals = np.dstack([np.transpose(predictions[:, :, i - 1] @ self._ns_mats, (1, 2, 0)), lagged_vals[:, :, :-r]])
            # Update the design matrix
            X = update_X(X, predictions[:, :, i-1], lagged_vals, self._p, self._s_vec)
            # Compute the (i + 1) - step ahead predictions
            predictions[:, :, i] = np.sum(X * coefficients, axis=2)
            if is_df:
                predictions_df.loc[:, (ts_names, i + 1)] = predictions[:, :, i] + mu
        
        # Return the predictions, adding the mean back to the data if necessary
        if is_df:
            return predictions_df
        return predictions + mu.reshape(1, n, 1)

    def simulate(self, m, sigma_2=None, burn_in=50):
        """
        Simulate data from the GNAR model.

        Parameters:
            m (int): The number of time steps to simulate.
            sigma_2 (int, float or np.ndarray): The variance of the noise. If an int or a float, the same variance is used for all time series.
            burn_in (int): The number of burn-in steps to discard.
        
        Returns:
            ts_sim (np.ndarray): The simulated time series data. Shape (m, n)
        """
        # Number of nodes and coefficients
        n = self._n
        r = np.max(self._s_vec)
        mu = self._parameters.iloc[0].to_numpy()
        coefficients = self._parameters.iloc[1:, :].to_numpy().T

        # Generate the noise - depending on the structure of sigma_2, different sampling methods are used. E.g, if sigma_2 is a scalar, the noise is sampled from np.random.normal, which is faster than using np.random.multivariate_normal using a diagonal covariance matrix.
        if sigma_2 is None:
            sigma_2 = self._sigma_2
        if isinstance(sigma_2, (float, int)):
            e_t = np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(burn_in + m, n))
        elif sigma_2.ndim == 1 or sigma_2.shape[0] == 1:
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
            ts_sim[t] = sim.copy()
            # Update neighbour sums array
            if self._p == 1:
                ns = (sim @ self._ns_mats).T
            else:
                ns = np.hstack([(sim @ self._ns_mats).T, ns[:, :-r]])
            # Shift the design matrix with the new simulated values and neighbour sums
            X = shift_X(X, sim, ns, self._p, self._s_vec)

        # Return the simulated time series data, adding the mean to the data
        return ts_sim[burn_in:] + mu

    def bic(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the GNAR model.

        Returns:
            bic (float): The Bayesian Information Criterion (BIC) for the GNAR model.
        """
        # Compute the BIC only if the model was fitted using time series data
        if self._m is None:
            raise ValueError("The model has not been fitted.")
        # Compute the log determinant of the noise covariance matrix
        if isinstance(self._sigma_2, (float, int)):
            det = n * np.log(self._sigma_2)
        elif self._sigma_2.ndim == 1 or self._sigma_2.shape[0] == 1:
            det = np.sum(np.log(self._sigma_2))
        else:
            det = np.log(np.linalg.det(self._sigma_2))
        # Compute the number of parameters in the model
        if self._model_type == "global":
            k = self._p + np.sum(self._s_vec)
        elif self._model_type == "standard":
            k = self._n * self._p + np.sum(self._s_vec)
        else:
            k = self._n * (self._p + np.sum(self._s_vec))
        # Compute the BIC
        return det + k * np.log(self._m - self._p) / (self._m - self._p)

    def aic(self):
        """
        Compute the Akaike Information Criterion (AIC) for the GNAR model.

        Returns:
            aic (float): The Akaike Information Criterion (AIC) for the GNAR model.
        """
        # Compute the AIC only if the model was fitted using time series data
        if self._m is None:
            raise ValueError("The model has not been fitted.")
        # Compute the log determinant of the noise covariance matrix
        if isinstance(self._sigma_2, (float, int)):
            det = n * np.log(self._sigma_2)
        elif self._sigma_2.ndim == 1 or self._sigma_2.shape[0] == 1:
            det = np.sum(np.log(self._sigma_2))
        else:
            det = np.log(np.linalg.det(self._sigma_2))
        # Compute the number of parameters in the model
        if self._model_type == "global":
            k = self._p + np.sum(self._s_vec)
        elif self._model_type == "standard":
            k = self._n * self._p + np.sum(self._s_vec)
        else:
            k = self._n * (self._p + np.sum(self._s_vec))
        # Compute the AIC
        return det + 2 * k / (self._m - self._p)
    
    def get_parameters(self):
        """
        Fetch the parameters of the model.
        """
        return self._parameters

    def get_covariance(self):
        """
        Fetch the covariance matrix of the model.
        """
        return self._cov
    
    def to_var(self):
        """
        Convert the GNAR model to VAR model format.
        
        Returns:
            var (VAR): The VAR model corresponding to the GNAR model.
        """
        # Get the mean and coefficients
        mu = self._parameters.iloc[0].to_numpy()
        coefficients = self._parameters.iloc[1:].to_numpy()
        var_coefficients = [mu]
        seen = 0
        # Construct a matrix for each lag of the model
        for i in range(self._p):
            # Alpha coefficients along the diagonal
            coeffs = np.diag(coefficients[i])
            s = self._s_vec[i]
            # Beta coefficients in the off-diagonal elements, reweighted according to the corresponding weights in the neighbour set matrices
            for j in range(s):
                coeffs += self._ns_mats[j] * coefficients[self._p + seen + j]
            seen += s
            var_coefficients.append(coeffs)
        # Create a parameter DataFrame for the VAR model with the same column names as the GNAR model
        var_parameters =  parameter_df(self._p, self._n, parameters=np.vstack(var_coefficients))
        var_parameters.columns = self._parameters.columns
        # Return the VAR model
        return VAR(p=self._p, parameters=var_parameters, sigma_2=self._sigma_2)

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
        """
        Return a string representation of the GNAR model.
        """
        model_info = f"{self._model_type.capitalize()} GNAR({self._p}, {self._s_vec}) Model\n"
        graph_info = f"{self.nx_graph}\n"
        parameter_info = f"Parameters:\n{self._parameters}\n"
        noise = f"Noise covariance matrix:\n{self._cov}\n"
        return model_info + graph_info + parameter_info + noise