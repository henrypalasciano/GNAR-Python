import numpy as np
import pandas as pd

from gnar.utils.simulating import generate_noise

class VAR:
    """
    Vector Autoregressive (VAR) model.

    Methods:
        fit: Fit the VAR model to the time series data.
        predict: Forecast future values of an input time series using the VAR model.
        simulate: Simulate data from the VAR model.
        bic: Compute the Bayesian Information Criterion (BIC) for the VAR model.
        aic: Compute the Akaike Information Criterion (AIC) for the VAR model.
        get_parameters: Fetch the parameters of the model.
        get_covariance: Fetch the covariance matrix of the model.
    """
    def __init__(self, p, ts=None, demean=True, coeffs=None, mean=1, sigma_2=1):
        """
        Initialise the VAR model.

        Parameters:
            p (int): The number of lags.
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of time series.
            demean (bool): Whether to remove the mean from the data. Only used if ts is provided.
            parameters (np.ndarray or pd.DataFrame): The parameters of the VAR model, consisting of the means and coefficients. Shape (1 + p * d, d).
            sigma_2 (float, int, np.ndarray or pd.DataFrame): The variance or covariance of the noise. If a float, the same variance is used for all time series. Only used if parameters is provided.
        """
        # Initial checks
        if p < 1:
            raise ValueError("The number of lags p must be at least 1.")

        self._p = p
        if ts is not None:
            # If a time series is provided, fit the model to the data, removing the mean if necessary
            self.fit(ts.copy(), demean)
        elif coeffs is not None:
            # If the parameters are provided, set up using these
            self._parameter_setup(coeffs, mean, sigma_2)        
        else:
            raise ValueError("Either the input time series data or the model parameters are required.")

    def _parameter_setup(self, coeffs, mean, sigma_2):
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
        if k != self._p * self._d:
            raise ValueError("The number of coefficients does not match the number of parameters.")
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
    
    def fit(self, ts, demean):
        """
        Fit the VAR model to the time series data.

        Parameters:
            ts (ndarray): The input time series data. Shape (n, d) where n is the number of observations and d is the number of time series.
            demean (bool): Whether to remove the mean from the data.
        """
        self._n, self._d = n, d = np.shape(ts)
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

        X = np.zeros((n - self._p, self._p * d))
        y = ts[self._p:]
        # Fill the design matrix with the lagged values
        for i in range(self._p):
            X[:, d * i : d * (i + 1)] = ts[self._p - i - 1 : n - i - 1, :]
        coeffs = np.zeros((d * self._p, d))
        for j in range(d):
            # Fit the model using least squares regression
            coeffs[:, j] = np.linalg.lstsq(X, y[:, j], rcond=None)[0]
        self._coeffs = coeffs
        
        # Compute the noise covariance matrix
        res = X @ coeffs - y
        self._sigma_2 = res.T @ res / (n - self._p)
        
    def predict(self, ts, h=1):
        """
        Forecast future values of an input time series using the VAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of nodes.
            h (int): The number of steps ahead to forecast.

        Returns:
            predictions (np.ndarray or pd.DataFrame): The predicted values. Shape (n + p - 1, d, h) if numpy array, or (n + p - 1, d * h) if DataFrame.
        """
        n, d = ts.shape
        if d != self._d:
            raise ValueError("The number of time series does not match the number of time series in the model.")

        # If the input is a DataFrame, create a DataFrame to store the predictions
        is_df = isinstance(ts, pd.DataFrame)
        if is_df:
            ts_names = ts.columns
            columns = pd.MultiIndex.from_product([ts_names, range(1, h + 1)], names=["Time Series", "Steps Ahead"])
            predictions_df = pd.DataFrame(index=ts.index[self._p - 1:], columns=columns, dtype=float)
            ts = ts.to_numpy()
        
        # Get the mean and coefficients
        mu = self._mu
        coeffs = self._coeffs

        ts = ts - mu
        # Fill the design matrix with the lagged values
        X = np.zeros((n - self._p + 1, self._p * d))
        for i in range(self._p):
            X[:, d * i : d * (i + 1)] = ts[self._p - i - 1 : n - i, :]

        # Compute the predictions
        predictions = np.zeros([n - self._p + 1, d, h])
        for i in range(h):
            predictions[:, :, i] = X @ coeffs
            # Update the design matrix
            if self._p == 1:
                X = predictions[:, :, i]
            else:
                X = np.hstack([predictions[:, :, i], X[:, :-d]])
            if is_df:
                predictions_df.loc[:, (ts_names, i + 1)] = predictions[:, :, i] + mu
                
        # Return the predictions, adding the mean back to the data if necessary
        if is_df:
            return predictions_df
        return predictions + mu.reshape(1, d, 1)

    def simulate(self, n, sigma_2=None, burn_in=50):
        """
        Simulate data from the VAR model.
        
        Parameters:
            n (int): The number of time steps to simulate.
            sigma_2 (int, float or np.ndarray): The variance of the noise. If an int or a float, the same variance is used for all time series.
            burn_in (int): The number of burn-in steps to discard.
        
        Returns:
            ts_sim (np.ndarray): The simulated time series data. Shape (n, d)
        """
        d = self._d
        mu = self._mu
        coeffs = self._coeffs.T

        # Generate the noise - depending on the structure of sigma_2, different sampling methods are used. E.g, if sigma_2 is a scalar, the noise is sampled from np.random.normal, which is faster than using np.random.multivariate_normal using a diagonal covariance matrix.
        if sigma_2 is None:
            sigma_2 = self._sigma_2
        e_t = generate_noise(sigma_2, burn_in + n, d)

        # Initialise the array to store the simulated time series data
        ts_sim = np.zeros([burn_in + n, d])
        X = np.zeros(self._p * d)
        for t in range(self._p, burn_in + n):
            # Compute the simulated observation
            sim = coeffs @ X + e_t[t]
            ts_sim[t] = sim.copy()
            # Update the design matrix
            if self._p == 1:
                X = sim
            else:
                X = np.hstack([sim, X[:-d]])
        # Return the simulated time series data, adding the mean to the data
        return ts_sim[burn_in:] + mu

    def bic(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the VAR model.

        Returns:
            bic (float): The Bayesian Information Criterion (BIC) for the VAR model.
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
        k = self._p * self._d * self._d
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
        k = self._p * self._d * self._d
        # Compute the AIC
        return det + 2 * k / (self._n - self._p)

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

    def __str__(self):
        """
        Return a string representation of the VAR model.
        """
        model_info = f"VAR({self._p}) Model\n"
        index = ["mean"] + [f"a_{i},{j}" for i in range(1, self._p + 1) for j in range(1, self._d + 1)]
        parameters = pd.DataFrame(np.vstack([self._mu, self._coeffs]), columns=self._names, index=index)
        parameters.columns.name = "ts"
        parameter_info = f"Parameters:\n{parameters}\n"
        if isinstance(self._sigma_2, (int, float)):
            cov = pd.DataFrame(self._sigma_2 * np.eye(self._d), index=self._names, columns=self._names)
        elif isinstance(self._sigma_2, np.ndarray) and (self._sigma_2.ndim == 1 or self._sigma_2.shape[0] == 1):
            cov =  pd.DataFrame(np.diag(self._sigma_2.flatten()), index=self._names, columns=self._names)
        else:
            cov = pd.DataFrame(self._sigma_2, index=self._names, columns=self._names)
        cov.columns.name = "ts"
        noise = f"Noise covariance matrix:\n{cov}\n"
        return model_info + parameter_info + noise