import numpy as np
import pandas as pd
import networkx as nx

from gnar.utils.model_fitting import gnar_lr
from gnar.utils.forecasting import format_X, update_X
from gnar.utils.simulating import shift_X, generate_noise
from gnar.utils.neighbour_sets import neighbour_set_mats
from gnar.var import VAR

class GNAR:
    """
    The GNAR model class for a d-dimensional mutivariate time series.

    Methods:
        fit: Fit the GNAR model to time series data.
        predict: Forecast future values of an input time series using the GNAR model.
        simulate: Simulate data from the GNAR model.
        bic: Compute the Bayesian Information Criterion (BIC) for the GNAR model.
        aic: Compute the Akaike Information Criterion (AIC) for the GNAR model.
        get_parameters: Fetch the parameters of the model.
        get_covariance: Fetch the covariance matrix of the noise.
        to_var: Convert the GNAR model to VAR model format.
        to_networkx: Convert the adjacency matrix to a NetworkX graph. 
        draw: Draw the graph using NetworkX.
    """
    def __init__(self, A, p, s, model_type="standard", ts=None, demean=True, coeffs=None, mean=0, sigma_2=1):
        """
        Initialise the GNAR model.

        Parameters:
            A (np.ndarray): The adjacency matrix of the graph.
            p (int): The number of lags.
            s (np.ndarray): An array containing the maximum stage of neighbour dependence for each lag.
            model_type (str): The type of GNAR model. Either "global", "standard" or "local". Defaults to "standard".
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of nodes.
            demean (bool): Whether to remove the mean from the data. Only required if ts is provided. Defaults to True.
            coeffs (np.ndarray or pd.DataFrame): The parameters of the GNAR model, consisting of the mean and coefficients of each node. Shape (1 + p + sum(s), d).
            mean (float, int, np.ndarray or pd.DataFrame): The mean of the time series data. If a float, the same mean is used for all nodes. Only required if parameters is provided. Defaults to 0.
            sigma_2 (float, int, np.ndarray or pd.DataFrame): The variance or covariance of the noise. If a float, the noise is assumed to have the same variance and be independent across nodes. Only required if parameters is provided. Defaults to 1.
        """
        # Initial checks
        if not isinstance(A, np.ndarray) or A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix A must be a square NumPy array.")
        if np.any(A < 0):
            raise ValueError("Adjacency matrix A must have non-negative weights.")
        if p < 1:
            raise ValueError("The number of lags p must be at least 1.")
        if not isinstance(s, np.ndarray) or len(s) != p:
            raise ValueError("Array s must be a NumPy array with length equal to p (number of lags).")
        if np.any(s < 0):
            raise ValueError("Values in s must be non-negative integers.")
        valid_model_types = {"global", "standard", "local"}
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type. Expected one of {valid_model_types}, got '{model_type}'.")

        self._A = A
        self._p = p
        self._s = s
        self._model_type = model_type
        # Compute the neighbour set matrices up to the maximum stage of neighbour dependence
        self._ns_mats = neighbour_set_mats(A, np.max(s))

        if ts is not None:
            # If a time series is provided, fit the model to the data, removing the mean if necessary
            self.fit(ts.copy(), demean)
        elif coeffs is not None:
            # If the parameters are provided, set up using these
            self._parameter_setup(model_type, coeffs, mean, sigma_2)    
            self._n = None    
        else:
            raise ValueError("Either the input time series data or the model parameters are required.")

    def _parameter_setup(self, model_type, coeffs, mean, sigma_2):
        # Store the coefficients
        if isinstance(coeffs, np.ndarray):
            self._coeffs = coeffs
            self._names = np.arange(1, np.shape(coeffs)[1] + 1)
        elif isinstance(coeffs, pd.DataFrame):
            self._coeffs = coeffs.to_numpy()
            self._names = coeffs.columns
        else:
            raise ValueError("Coefficients must be a NumPy array or a Pandas DataFrame.")
        # Check the dimensions of the coefficients matrix
        k, self._d = np.shape(self._coeffs)
        if self._d != A.shape[0]:
            raise ValueError("The number of nodes in the adjacency matrix does not match the number of nodes in the coefficients matrix.")
        if k != self._p + np.sum(self._s):
            raise ValueError("The number of coefficients does not match the number of parameters.")
        unique_params = np.apply_along_axis(lambda row: len(np.unique(row)), axis=1, arr=self._coeffs)
        if self._model_type == "global" and np.any(unique_params != 1):
            raise ValueError("The model type is global, but the coefficients are not the same for all nodes.")
        elif self._model_type == "standard" and np.any(unique_params[self._d * self._p:] != 1):
            raise ValueError("The model type is standard, but the beta coefficients are not the same for all nodes.")
        # Store the mean
        if isinstance(mean, (float, int)):
            self._mu = np.repeat(mean, self._d).reshape(1, self._d)
        elif isinstance(mean, np.ndarray):
            self._mu = mean.reshape(1, self._d)
        elif isinstance(mean, pd.DataFrame):
            self._mu = mean.to_numpy().reshape(1, self._d)
        else:
            raise ValueError("Mean must be a float, int, NumPy array or Pandas DataFrame.")
        # Store the noise covariance matrix
        if isinstance(sigma_2, (float, int, np.ndarray)):
            self._sigma_2 = sigma_2
        elif isinstance(sigma_2, pd.DataFrame):
            self._sigma_2 = sigma_2.to_numpy()
        else:
            raise ValueError("Noise covariance matrix must be a float, int, NumPy array or Pandas DataFrame.")

    def fit(self, ts, demean=True):
        """
        Fit the GNAR model to the time series data.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of nodes.
            demean (bool): Whether to remove the mean from the data.
        """
        self._n, self._d = np.shape(ts)
        if isinstance(ts, pd.DataFrame):
            self._names = ts.columns
            ts = ts.to_numpy()
        else:
            self._names = np.arange(1, self._d + 1)
        # Compute the mean if necessary and save this to the parameter DataFrame
        if demean:
            self._mu = mu = np.mean(ts, axis=0, keepdims=True)
        else:
            self._mu = mu = np.zeros((1, self._d))
        ts -= mu

        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (n, d, 1 + r), where r = max(s)
        data = np.zeros([self._n, self._d, 1 + np.max(self._s)])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._ns_mats, (1, 2, 0))
        # Fit the model using least squares linear regression
        coeffs, sigma_2 = gnar_lr(data, self._p, self._s, self._model_type)
        self._coeffs = coeffs.T
        self._sigma_2 = sigma_2

    def predict(self, ts, h=1):
        """
        Forecast future values of an input time series using the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of nodes.
            h (int): The number of steps ahead to forecast.

        Returns:
            predictions (np.ndarray or pd.DataFrame): The predicted values. Shape (n + p - 1, d, h) if numpy array, or (n + p - 1, d * h) if DataFrame.
        """
        # Data shapes
        n, d = ts.shape
        r = np.max(self._s)
        if d != self._d:
            raise ValueError("The number of time series does not match the number of nodes in the model.")

        # If the input is a DataFrame, create a DataFrame to store the predictions
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            ts_names = ts.columns
            columns = pd.MultiIndex.from_product([ts_names, range(1, h + 1)], names=["Time Series", "Steps Ahead"])
            predictions_df = pd.DataFrame(index=ts.index[self._p - 1:], columns=columns, dtype=float)
            ts = ts.to_numpy()
        
        # Get the mean and coefficients
        mu = self._mu
        coeffs = self._coeffs.T
        
        ts = ts - mu
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (n, d, 1 + r)
        data = np.zeros([n, d, 1 + r])
        data[:, :, 0] = ts
        data[:, :, 1:] = np.transpose(ts @ self._ns_mats, (1, 2, 0))
        # Format the data to make predictions. The laggad values array contains lags for the time series and neighbour sums up to the maximum neighbour stage of dependence. This is an array of shape (n - p + 1, d, p * (1 + r)), which is required for updating X.
        X, lagged_vals = format_X(data, self._p, self._s)

        # Initialise the array to store the predictions, which is an array of shape (n - p + 1, d, h)
        predictions = np.zeros([n - self._p + 1, d, h])
        # Compute the one-step ahead predictions
        predictions[:, :, 0] = np.sum(X * coeffs, axis=2)
        if is_df:
            predictions_df.loc[:, (ts_names, 1)] = predictions[:, :, 0] + mu
        for i in range(1, h):
            # Update the lagged values and design matrix using the predicted values
            lagged_vals = np.dstack([np.transpose(predictions[:, :, i - 1] @ self._ns_mats, (1, 2, 0)), lagged_vals[:, :, :-r]])
            # Update the design matrix
            X = update_X(X, predictions[:, :, i-1], lagged_vals, self._p, self._s)
            # Compute the (i + 1) - step ahead predictions
            predictions[:, :, i] = np.sum(X * coeffs, axis=2)
            if is_df:
                predictions_df.loc[:, (ts_names, i + 1)] = predictions[:, :, i] + mu
        
        # Return the predictions, adding the mean back to the data if necessary
        if is_df:
            return predictions_df
        return predictions + mu.reshape(1, d, 1)

    def simulate(self, n, sigma_2=None, burn_in=50):
        """
        Simulate data from the GNAR model.

        Parameters:
            n (int): The number of time steps to simulate.
            sigma_2 (int, float or np.ndarray): The variance of the noise. If an int or a float, the same variance is used for all time series.
            burn_in (int): The number of burn-in steps to discard.
        
        Returns:
            ts_sim (np.ndarray): The simulated time series data. Shape (n, d)
        """
        # Number of nodes and coefficients
        d = self._d
        r = np.max(self._s)
        mu = self._mu
        coeffs = self._coeffs.T

        # Generate the noise - depending on the structure of sigma_2, different sampling methods are used. E.g, if sigma_2 is a scalar, the noise is sampled from np.random.normal, which is faster than using np.random.multivariate_normal using a diagonal covariance matrix.
        if sigma_2 is None:
            sigma_2 = self._sigma_2
        e_t = generate_noise(sigma_2, burn_in + n, d)
        
        # Initialise the array to store the simulated time series data
        ts_sim = np.zeros([burn_in + n, d])
        ns = np.zeros([d, self._p * r])
        # Initialise the design matrix, which is an array of shape (d, p + sum(s))
        X = np.zeros([d, self._p + np.sum(self._s)])
        for t in range(self._p, burn_in + n):
            # Compute the simulated observation
            sim = np.sum(X * coeffs, axis=1) + e_t[t]
            ts_sim[t] = sim.copy()
            # Update neighbour sums array
            if self._p == 1:
                ns = (sim @ self._ns_mats).T
            else:
                ns = np.hstack([(sim @ self._ns_mats).T, ns[:, :-r]])
            # Shift the design matrix with the new simulated values and neighbour sums
            X = shift_X(X, sim, ns, self._p, self._s)

        # Return the simulated time series data, adding the mean to the data
        return ts_sim[burn_in:] + mu

    def bic(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the GNAR model.

        Returns:
            bic (float): The Bayesian Information Criterion (BIC) for the GNAR model.
        """
        # Compute the BIC only if the model was fitted using time series data
        if self._n is None:
            raise ValueError("The model was not fit.")
        # Compute the log determinant of the noise covariance matrix
        if isinstance(self._sigma_2, (float, int)):
            det = self._d * np.log(self._sigma_2)
        elif self._sigma_2.ndim == 1 or self._sigma_2.shape[0] == 1:
            det = np.sum(np.log(self._sigma_2))
        else:
            det = np.log(np.linalg.det(self._sigma_2))
        # Compute the number of parameters in the model
        if self._model_type == "global":
            k = self._p + np.sum(self._s)
        elif self._model_type == "standard":
            k = self._d * self._p + np.sum(self._s)
        else:
            k = self._d * (self._p + np.sum(self._s))
        # Compute the BIC
        return det + k * np.log(self._n - self._p) / (self._n - self._p)

    def aic(self):
        """
        Compute the Akaike Information Criterion (AIC) for the GNAR model.

        Returns:
            aic (float): The Akaike Information Criterion (AIC) for the GNAR model.
        """
        # Compute the AIC only if the model was fitted using time series data
        if self._n is None:
            raise ValueError("The model was not fit.")
        # Compute the log determinant of the noise covariance matrix
        if isinstance(self._sigma_2, (float, int)):
            det = self._d * np.log(self._sigma_2)
        elif self._sigma_2.ndim == 1 or self._sigma_2.shape[0] == 1:
            det = np.sum(np.log(self._sigma_2))
        else:
            det = np.log(np.linalg.det(self._sigma_2))
        # Compute the number of parameters in the model
        if self._model_type == "global":
            k = self._p + np.sum(self._s)
        elif self._model_type == "standard":
            k = self._d * self._p + np.sum(self._s)
        else:
            k = self._d * (self._p + np.sum(self._s))
        # Compute the AIC
        return det + 2 * k / (self._n - self._p)
    
    def get_coeffs(self):
        """
        Fetch the coefficients of the model.
        """
        return self._coeffs

    def get_mean(self):
        """
        Fetch the means of the time series.
        """
        return self._mu

    def get_cov(self):
        """
        Fetch the covariance matrix of the noise.
        """
        return self._sigma_2
    
    def to_var(self):
        """
        Convert the GNAR model to VAR model format.
        
        Returns:
            var (VAR): The VAR form of the GNAR model.
        """
        # Get coefficients
        coefficients = self._coeffs
        var_coeffs = []
        seen = 0
        # Construct a matrix for each lag of the model
        for i in range(self._p):
            # Alpha coefficients along the diagonal
            coeffs = np.diag(coefficients[i])
            s = self._s[i]
            # Beta coefficients in the off-diagonal elements, reweighted according to the corresponding weights in the neighbour set matrices
            for j in range(s):
                coeffs += self._ns_mats[j] * coefficients[self._p + seen + j]
            seen += s
            var_coeffs.append(coeffs)
        # Create a parameter DataFrame for the VAR model with the same column names as the GNAR model
        var_coeffs =  pd.DataFrame(np.vstack(var_coeffs), columns=self._names)
        # Return the VAR model
        return VAR(p=self._p, coeffs=var_coeffs, mean=self._mu, sigma_2=self._sigma_2)

    def to_networkx(self):
        """
        Convert the adjacency matrix to a NetworkX graph, which is stored in the nx_graph attribute.
        """
        nx_graph = nx.from_numpy_array(self._A)
        return nx.relabel_nodes(nx_graph, dict(enumerate(self._names)))

    def draw(self):
        """
        Draw the graph using NetworkX.
        """
        nx_graph = self.to_networkx()
        nx.draw(nx_graph, with_labels=True)

    def __str__(self):
        """
        Return a string representation of the GNAR model.
        """
        model_info = f"{self._model_type.capitalize()} GNAR({self._p}, {self._s}) Model\n"
        nx_graph = self.to_networkx()
        graph_info = f"{nx_graph}\n"
        index = ["mean"] + [f"a_{i}" for i in range(1, self._p + 1)]
        for i in range(1, self._p + 1):
            index += [f"b_{i},{j}" for j in range(1, self._s[i - 1] + 1)]
        parameters = pd.DataFrame(np.vstack([self._mu, self._coeffs]), columns=self._names, index=index)
        parameters.columns.name = "node"
        parameter_info = f"Parameters:\n{parameters}\n"
        if isinstance(self._sigma_2, (int, float)):
            cov = pd.DataFrame(self._sigma_2 * np.eye(self._d), index=self._names, columns=self._names)
        elif isinstance(self._sigma_2, np.ndarray) and (self._sigma_2.ndim == 1 or self._sigma_2.shape[0] == 1):
            cov =  pd.DataFrame(np.diag(self._sigma_2.flatten()), index=self._names, columns=self._names)
        else:
            cov = pd.DataFrame(self._sigma_2, index=self._names, columns=self._names)
        cov.columns.name = "node"
        noise = f"Noise covariance matrix:\n{cov}\n"
        return model_info + graph_info + parameter_info + noise