import numpy as np
import pandas as pd
import networkx as nx

from gnar.utils.gnar_linear_regression import gnar_lr
from gnar.utils.forecasting import format_X, update_X
from gnar.utils.simulating import shift_X, generate_noise
from gnar.utils.neighbour_sets import *
from gnar.utils.data_utils import *
from gnar.var import VAR

class GNAR:
    """
    Generalised Network Autoregressive (GNAR) model. Can be initialiased by either fitting the model to time series data or providing the model parameters.

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

    Methods:
        fit: Fit the GNAR model to time series data.
        predict: Forecast future values of an input time series using the GNAR model.
        simulate: Simulate a realisation from the GNAR model.
        bic: Compute the Bayesian Information Criterion (BIC).
        aic: Compute the Akaike Information Criterion (AIC).
        to_var: Convert the GNAR model to VAR model format.
        to_networkx: Convert the adjacency matrix to a NetworkX graph. 
        draw: Draw the graph using NetworkX.
    """
    def __init__(self, A, p, s, model_type="standard", ts=None, demean=True, coeffs=None, mean=0, sigma_2=1):
        # Initial checks
        gnar_checks(A, p, s, model_type)

        self._A = A
        self._p = p
        self._s = s
        self._model_type = model_type
        # Compute the neighbour set matrices up to the maximum stage of neighbour dependence
        if np.all((A == 0) | (A == 1)):
            self._ns_mats = neighbour_set_mats(A, np.max(s))
        else:
            raise ValueError("Weighted matrices not supported yet.")

        if ts is not None:
            # If a time series is provided, fit the model to the data, removing the mean if necessary
            self.fit(ts.copy(), demean)
        elif coeffs is not None:
            # If the parameters are provided, set up using these
            self._d = A.shape[0]
            self._n = None 
            self._ts = None
            self._parameter_setup(coeffs, mean, sigma_2)    
        else:
            raise ValueError("Either the input time series data or the model parameters are required.")

    def _parameter_setup(self, coeffs, mean, sigma_2):
        # Store the coefficients
        if isinstance(coeffs, np.ndarray):
            self.coeffs = coeffs
            self._names = np.arange(1, np.shape(coeffs)[1] + 1)
        elif isinstance(coeffs, pd.DataFrame):
            self.coeffs = coeffs.to_numpy()
            self._names = coeffs.columns
        else:
            raise ValueError("Coefficients must be a NumPy array or a Pandas DataFrame.")
        # Check the dimensions and structure of the coefficients matrix
        check_gnar_coeffs(self.coeffs, self._d, self._p, self._s, self._model_type)
        # Store the mean of the time series data and the covariance matrix of the noise
        self.mu = set_mean(mean, self._d)
        self.sigma_2 = set_cov(sigma_2)

    def fit(self, ts, demean=True):
        """
        Fit the GNAR model to the time series data.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of nodes.
            demean (bool): Whether to remove the mean from the data.
        """
        self._n, self._d = np.shape(ts)
        self._ts = ts
        if isinstance(ts, pd.DataFrame):
            self._names = ts.columns
            ts = ts.to_numpy()
        else:
            self._names = np.arange(1, self._d + 1)
        # Compute the mean if necessary and save this to the parameter DataFrame
        if demean:
            self.mu = np.mean(ts, axis=0, keepdims=True)
            ts = ts - self.mu
        else:
            self.mu = np.zeros((1, self._d))

        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (n, d, 1 + r), where r = max(s)
        data = compute_neighbour_sums(ts, self._ns_mats, np.max(self._s))
        # Fit the model using least squares linear regression
        coeffs, sigma_2 = gnar_lr(data, self._p, self._s, self._model_type)
        self.coeffs = coeffs.T
        self.sigma_2 = sigma_2

    def predict(self, ts=None, h=1):
        """
        Forecast future values of an input time series using the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of nodes. If None, the model the forecasts from the last observation used in fitting.
            h (int): The number of steps ahead to forecast.

        Returns:
            preds (np.ndarray or pd.DataFrame): The predicted values. If the shape of the input is (p, d), the shape of the output is always (h, d). If 
                                                the shape is (n, d) for some n > p, the output is (n - p + 1, d, h) if a numpy array, or (n - p + 1, d * h) 
                                                if a pandas DataFrame. In the latter case we are assuming that one computes forecasts from each available
                                                time point, which may be useful when evaluating the performance of a model out-of-sample for example.
        """
        if ts is None:
            if self._n is None:
                raise ValueError("The model was not fit.")
            # Last p observations used in fitting
            ts = self._ts[-self._p:]
        # Data shapes
        n, d = ts.shape
        r = np.max(self._s)
        if d != self._d:
            raise ValueError("The number of time series does not match the number of nodes in the model.")
        if n < self._p:
            raise ValueError("The number of observations is insufficient for forecasting.")

        # DataFrame handling
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            names = ts.columns
            index = ts.index[self._p - 1:]
            ts = ts.to_numpy()
        
        # Remove the mean from the data and get the coefficients
        ts = ts - self.mu
        coeffs = self.coeffs.T
        # Compute the neighbour sums up to the maximum stage of neighbour dependence. This is an array of shape (n, d, 1 + r)
        data = compute_neighbour_sums(ts, self._ns_mats, r)
        # Format the data to make predictions. The laggad values array contains lags for the time series and neighbour sums up to the maximum neighbour stage of dependence. This is an array of shape (n - p + 1, d, p * (1 + r)), which is required for updating X.
        X, lagged_vals = format_X(data, self._p, self._s)

        # Initialise the array to store the predictions, which is an array of shape (n - p + 1, d, h)
        preds = np.zeros([n - self._p + 1, d, h])
        # Compute the one-step ahead predictions
        preds[:, :, 0] = np.sum(X * coeffs, axis=2)
        for i in range(1, h):
            # Update the lagged values and design matrix using the predicted values
            lagged_vals = np.dstack([np.transpose(preds[:, :, i - 1] @ self._ns_mats, (1, 2, 0)), lagged_vals[:, :, :-r]])
            # Update the design matrix
            X = update_X(X, preds[:, :, i-1], lagged_vals, self._p, self._s)
            # Compute the (i + 1) - step ahead predictions
            preds[:, :, i] = np.sum(X * coeffs, axis=2)
            
        preds = preds + self.mu.reshape(1, d, 1)
        if n == self._p:
            if is_df:
                return pd.DataFrame(preds[0].T, index=range(1, h + 1), columns=names, dtype=float)
            return preds[0].T
        if is_df:
            columns = pd.MultiIndex.from_product([names, [f"h = {i}" for i in range(1, h + 1)]])
            return pd.DataFrame(preds.reshape(n - self._p + 1, d * h), index=index, columns=columns, dtype=float)
        return preds

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
        coeffs = self.coeffs.T

        # Generate the noise - depending on the structure of sigma_2, different sampling methods are used. E.g, if sigma_2 is a scalar, the noise is sampled from np.random.normal, which is faster than using np.random.multivariate_normal using a diagonal covariance matrix.
        if sigma_2 is None:
            sigma_2 = self.sigma_2
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
        return ts_sim[burn_in:] + self.mu

    def compute_autocov_mats(self, max_lag=None):
        """
        Compute the autocovariance matrices for the GNAR model up to a maximum lag. Output shape: (max_lag + 1, d, d) from lag 0 to lag max_lag
        """
        # Convert to a VAR
        var = self.to_var()
        # Compute the autocovariance matrices
        return var.compute_autocov_mats(max_lag=max_lag)

    def bic(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the GNAR model.
        """
        # Compute the BIC only if the model was fitted using time series data
        if self._n is None:
            raise ValueError("The model was not fit.")
        det = np.log(np.linalg.det(self.sigma_2))
        k = self._num_params()
        # Compute the BIC
        return det + k * np.log(self._n - self._p) / (self._n - self._p)

    def aic(self):
        """
        Compute the Akaike Information Criterion (AIC) for the GNAR model.
        """
        # Compute the AIC only if the model was fitted using time series data
        if self._n is None:
            raise ValueError("The model was not fit.")
        det = np.log(np.linalg.det(self.sigma_2))
        k = self._num_params()
        # Compute the AIC
        return det + 2 * k / (self._n - self._p)
    
    def _num_params(self):
        # Compute the number of parameters in the model
        if self._model_type == "global":
            return self._p + np.sum(self._s)
        elif self._model_type == "standard":
            return self._d * self._p + np.sum(self._s)
        return self._d * (self._p + np.sum(self._s))
    
    def to_var(self):
        """
        Convert the GNAR model to VAR model format.
        
        Returns:
            var (VAR): The VAR form of the GNAR model.
        """
        # Get coefficients
        coefficients = self.coeffs
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
        return VAR(p=self._p, coeffs=var_coeffs, mean=self.mu, sigma_2=self.sigma_2)

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
        parameters = pd.DataFrame(np.vstack([self.mu, self.coeffs]), columns=self._names, index=index)   
        parameter_info = f"Parameters:\n{parameters}\n"
        cov = pd.DataFrame(cov_mat(self.sigma_2, self._d), index=self._names, columns=self._names)
        noise = f"Noise covariance matrix:\n{cov}\n"
        return model_info + graph_info + parameter_info + noise