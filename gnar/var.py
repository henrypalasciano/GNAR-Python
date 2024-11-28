import numpy as np
import pandas as pd

from gnar.utils.formatting import parameter_df, cov_df
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
    def __init__(self, p, ts=None, remove_mean=True, parameters=None, sigma_2=1):
        """
        Initialise the VAR model.

        Parameters:
            p (int): The number of lags.
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of time series.
            remove_mean (bool): Whether to remove the mean from the data. Only used if ts is provided.
            parameters (np.ndarray or pd.DataFrame): The parameters of the VAR model, consisting of the means and coefficients. Shape (1 + p * d, d).
            sigma_2 (float, int, np.ndarray or pd.DataFrame): The variance or covariance of the noise. If a float, the same variance is used for all time series. Only used if parameters is provided.
        """
        self._p = p
        self._n = None
        self._d = None
        self._model_setup(ts, remove_mean, parameters, sigma_2)

    def _model_setup(self, ts, remove_mean, parameters, sigma_2):
        """
        Setup the GNAR model.

        Parameters:
            ts (np.ndarray or pd.DataFrame): The input time series data. Shape (n, d) where n is the number of observations and d is the number of time series.
            remove_mean (bool): Whether to remove the mean from the data. Only used if ts is provided.
            parameters (np.ndarray or pd.DataFrame): The parameters of the VAR model, consisting of the means and coefficients. Shape (1 + p * d, d).
            sigma_2 (float, int, np.ndarray or pd.DataFrame): The variance or covariance of the noise. If a float, the same variance is used for all time series. Only used if parameters is provided.
        """
        # If a time series is provided, fit the model to the data, removing the mean if necessary
        if ts is not None:
            self._n, self._d = ts.shape
            self._parameters = parameter_df(self._p, ts.shape[1], ts=ts)
            if isinstance(ts, np.ndarray):
                self.fit(ts, remove_mean)
            else:
                self.fit(ts.to_numpy(), remove_mean)
        # If the parameters are provided, set the parameters and covariance matrix
        elif parameters is not None:
            self._d = parameters.shape[1]
            if isinstance(parameters, pd.DataFrame):
                self._parameters = parameters
            else:
                self._parameters = parameter_df(self._p, self._d, parameters=parameters)
            if isinstance(sigma_2, (float, int, np.ndarray)):
                self._sigma_2 = sigma_2
            else:
                self._sigma_2 = sigma_2.to_numpy()
            self._cov = cov_df(sigma_2, self._parameters.columns)
        else:
            raise ValueError("Either the input time series data or the parameters are required.")
    
    def fit(self, ts, remove_mean):
        """
        Fit the VAR model to the time series data.

        Parameters:
            ts (ndarray): The input time series data. Shape (n, d) where n is the number of observations and d is the number of time series.
            remove_mean (bool): Whether to remove the mean from the data.
        """
        n, d = self._n, self._d
        # Compute the mean if necessary and save this to the parameter DataFrame
        if remove_mean:
            mu = np.mean(ts, axis=0, keepdims=True)
        else:
            mu = np.zeros((1, d))
        ts = ts - mu
        self._parameters.loc["mean"] = mu

        X = np.zeros((n - self._p, self._p * d))
        y = ts[self._p:]
        # Fill the design matrix with the lagged values
        for i in range(self._p):
            X[:, d * i : d * (i + 1)] = ts[self._p - i - 1 : n - i - 1, :]
        coefficients = np.zeros((d * self._p, d))
        for j in range(d):
            # Fit the model using least squares regression
            coefficients[:, j] = np.linalg.lstsq(X, y[:, j], rcond=None)[0]
        self._parameters.iloc[1:] = coefficients
        
        # Compute the noise covariance matrix
        res = X @ coefficients - y
        self._sigma_2 = res.T @ res / (n - self._p)
        # Store the covariance matrix as a DataFrame for display purposes
        self._cov = pd.DataFrame(self._sigma_2, index=self._parameters.columns, columns=self._parameters.columns, dtype=float)
    
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
        mu = self._parameters.to_numpy()[0]
        coefficients = self._parameters.to_numpy()[1:]

        ts = ts - mu
        # Fill the design matrix with the lagged values
        X = np.zeros((n - self._p + 1, self._p * d))
        for i in range(self._p):
            X[:, d * i : d * (i + 1)] = ts[self._p - i - 1 : n - i, :]

        # Compute the predictions
        predictions = np.zeros([n - self._p + 1, d, h])
        for i in range(h):
            predictions[:, :, i] = X @ coefficients
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
        mu = self._parameters.iloc[0].to_numpy()
        coefficients = self._parameters.iloc[1:, :].to_numpy().T

        # Generate the noise - depending on the structure of sigma_2, different sampling methods are used. E.g, if sigma_2 is a scalar, the noise is sampled from np.random.normal, which is faster than using np.random.multivariate_normal using a diagonal covariance matrix.
        if sigma_2 is None:
            sigma_2 = self._sigma_2
        e_t = generate_noise(sigma_2, burn_in + n, d)

        # Initialise the array to store the simulated time series data
        ts_sim = np.zeros([burn_in + n, d])
        X = np.zeros(self._p * d)
        for t in range(self._p, burn_in + n):
            # Compute the simulated observation
            sim = coefficients @ X + e_t[t]
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
            raise ValueError("The model has not been fitted.")
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
            raise ValueError("The model has not been fitted.")
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
        parameter_info = f"Parameters:\n{self._parameters}\n"
        noise = f"Noise covariance matrix:\n{self._cov}\n"
        return model_info + parameter_info + noise