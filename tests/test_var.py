import numpy as np
import pandas as pd
import pytest

from gnar import VAR


class TestVARFit:
    """Tests for VAR model fitting."""

    def test_fit_coeff_shape(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        # coeffs shape: (p * d, d) = (6, 3)
        assert V.coeffs.shape == (6, 3)

    def test_fit_dataframe(self, ts_df):
        V = VAR(p=2, ts=ts_df)
        assert V.coeffs.shape == (6, 3)

    def test_sigma2_shape(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        assert V.sigma_2.shape == (3, 3)

    def test_fit_p1(self, ts_np):
        V = VAR(p=1, ts=ts_np)
        assert V.coeffs.shape == (3, 3)


class TestVARPredict:
    """Tests for VAR prediction."""

    def test_predict_default(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        preds = V.predict(h=3)
        assert preds.shape == (3, 3)

    def test_predict_single_window(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        preds = V.predict(ts=ts_np[:2], h=5)
        assert preds.shape == (5, 3)

    def test_predict_batch(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        preds = V.predict(ts=ts_np[:10], h=2)
        assert preds.shape == (9, 3, 2)

    def test_predict_no_ts_coeffs_only(self):
        coeffs = 0.1 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit to data"):
            V.predict()

    def test_predict_dataframe_output(self, ts_df):
        V = VAR(p=2, ts=ts_df)
        preds = V.predict(ts=ts_df.iloc[:2], h=3)
        assert isinstance(preds, pd.DataFrame)


class TestVARSimulate:
    """Tests for VAR simulation."""

    def test_simulate_shape(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        sim = V.simulate(200)
        assert sim.shape == (200, 3)

    def test_simulate_custom_sigma(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        sim = V.simulate(100, sigma_2=0.5)
        assert sim.shape == (100, 3)


class TestVARStationarity:
    """Tests for stationarity check."""

    def test_stationary_model(self):
        coeffs = 0.3 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        assert V.is_stationary()

    def test_non_stationary_model(self):
        coeffs = 2.0 * np.eye(3)
        with pytest.warns(UserWarning, match="non-stationary"):
            V = VAR(p=1, coeffs=coeffs)
        assert not V.is_stationary()


class TestVARCompanion:
    """Tests for companion form."""

    def test_companion_p1(self, ts_np):
        V = VAR(p=1, ts=ts_np)
        phi, sigma = V.companion_form()
        assert phi.shape == (3, 3)
        assert sigma.shape == (3, 3)

    def test_companion_p2(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        phi, sigma = V.companion_form()
        assert phi.shape == (6, 6)
        assert sigma.shape == (6, 6)


class TestVARInfoCriteria:
    """Tests for BIC and AIC."""

    def test_bic_returns_float(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        bic = V.bic()
        assert isinstance(bic, (float, np.floating))

    def test_aic_returns_float(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        aic = V.aic()
        assert isinstance(aic, (float, np.floating))

    def test_bic_not_fitted(self):
        coeffs = 0.1 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit"):
            V.bic()

    def test_aic_not_fitted(self):
        coeffs = 0.1 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit"):
            V.aic()


class TestVARRepr:
    """Tests for __repr__ and __str__."""

    def test_repr_fitted(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        r = repr(V)
        assert "VAR(" in r
        assert "fitted=True" in r

    def test_repr_not_fitted(self):
        coeffs = 0.1 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        r = repr(V)
        assert "fitted=False" in r

    def test_str(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        s = str(V)
        assert "VAR" in s
        assert "Parameters" in s


class TestVARAutocov:
    """Tests for autocovariance/autocorrelation computation."""

    def test_autocov_shape(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        autocovs = V.compute_autocov_mats(max_lag=5)
        # Shape: (max_lag + 1, d, d)
        assert autocovs.shape == (6, 3, 3)

    def test_autocorr_shape(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        autocorrs = V.compute_autocorr_mats(max_lag=5)
        assert autocorrs.shape == (6, 3, 3)

    def test_autocorr_lag0_diagonal(self, ts_np):
        V = VAR(p=2, ts=ts_np)
        autocorrs = V.compute_autocorr_mats(max_lag=3)
        # Lag 0 autocorrelation diagonal should be all 1s
        assert np.allclose(np.diag(autocorrs[0]), 1.0)
