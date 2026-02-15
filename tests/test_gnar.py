import numpy as np
import pandas as pd
import pytest

from gnar import GNAR


class TestGNARFit:
    """Tests for GNAR model fitting."""

    def test_fit_ols_coeff_shape(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, method="OLS")
        # coeffs shape: (p + sum(s), d) = (2 + 2, 3) = (4, 3)
        assert G.coeffs.shape == (4, 3)

    def test_fit_yw_coeff_shape(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, method="YW")
        assert G.coeffs.shape == (4, 3)

    def test_fit_global(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, model_type="global")
        # All columns should be identical for global model
        assert np.allclose(G.coeffs[:, 0], G.coeffs[:, 1])
        assert np.allclose(G.coeffs[:, 0], G.coeffs[:, 2])

    def test_fit_standard(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, model_type="standard")
        # Beta rows (rows 2 and 3) should be identical across columns
        assert np.allclose(G.coeffs[2, 0], G.coeffs[2, 1])
        assert np.allclose(G.coeffs[3, 0], G.coeffs[3, 1])

    def test_fit_local(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, model_type="local")
        assert G.coeffs.shape == (4, 3)

    def test_fit_dataframe(self, adjacency_3, ts_df, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_df, method="OLS")
        assert G.coeffs.shape == (4, 3)

    def test_sigma2_shape(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        assert G.sigma_2.shape == (3, 3)


class TestGNARPredict:
    """Tests for GNAR prediction."""

    def test_predict_default(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        preds = G.predict(h=3)
        # Default: uses last p obs, output shape (h, d) = (3, 3)
        assert preds.shape == (3, 3)

    def test_predict_single_window(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        preds = G.predict(ts=ts_np[:2], h=5)
        assert preds.shape == (5, 3)

    def test_predict_batch(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        preds = G.predict(ts=ts_np[:10], h=2)
        # Shape: (n - p + 1, d, h) = (9, 3, 2)
        assert preds.shape == (9, 3, 2)

    def test_predict_dataframe_output(self, adjacency_3, ts_df, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_df)
        preds = G.predict(ts=ts_df.iloc[:2], h=3)
        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (3, 3)

    def test_predict_no_ts_coeffs_only(self, adjacency_3, s_array):
        coeffs = np.zeros((4, 3))
        G = GNAR(adjacency_3, p=2, s=s_array, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit to data"):
            G.predict()

    def test_predict_wrong_d(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        with pytest.raises(ValueError):
            G.predict(ts=np.random.normal(0, 1, (10, 5)))


class TestGNARSimulate:
    """Tests for GNAR simulation."""

    def test_simulate_shape(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        sim = G.simulate(200)
        assert sim.shape == (200, 3)

    def test_simulate_mean_stationarity(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, demean=True)
        np.random.seed(0)
        sim = G.simulate(5000, burn_in=200)
        # Mean should be close to the fitted mean
        assert np.allclose(np.mean(sim, axis=0), G.mu.flatten(), atol=0.3)


class TestGNARInfoCriteria:
    """Tests for BIC and AIC."""

    def test_bic_returns_float(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        bic = G.bic()
        assert isinstance(bic, (float, np.floating))

    def test_aic_returns_float(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        aic = G.aic()
        assert isinstance(aic, (float, np.floating))

    def test_bic_not_fitted(self, adjacency_3, s_array):
        coeffs = np.zeros((4, 3))
        G = GNAR(adjacency_3, p=2, s=s_array, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit"):
            G.bic()

    def test_aic_not_fitted(self, adjacency_3, s_array):
        coeffs = np.zeros((4, 3))
        G = GNAR(adjacency_3, p=2, s=s_array, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit"):
            G.aic()


class TestGNARConversion:
    """Tests for to_var conversion."""

    def test_to_var_type(self, adjacency_3, ts_np, s_array):
        from gnar import VAR
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        V = G.to_var()
        assert isinstance(V, VAR)

    def test_to_var_coeffs_shape(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        V = G.to_var()
        # VAR coeffs shape: (p * d, d) = (6, 3)
        assert V.coeffs.shape == (6, 3)

    def test_to_var_predict_consistency(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        V = G.to_var()
        # 1-step predictions should be identical
        gnar_pred = G.predict(ts=ts_np[-2:], h=1)
        var_pred = V.predict(ts=ts_np[-2:], h=1)
        assert np.allclose(gnar_pred, var_pred, atol=1e-10)


class TestGNARRepr:
    """Tests for __repr__ and __str__."""

    def test_repr_fitted(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        r = repr(G)
        assert "GNAR(" in r
        assert "fitted=True" in r

    def test_repr_not_fitted(self, adjacency_3, s_array):
        coeffs = np.zeros((4, 3))
        G = GNAR(adjacency_3, p=2, s=s_array, coeffs=coeffs)
        r = repr(G)
        assert "fitted=False" in r

    def test_str(self, adjacency_3, ts_np, s_array):
        G = GNAR(adjacency_3, p=2, s=s_array, ts=ts_np)
        s = str(G)
        assert "GNAR" in s
        assert "Parameters" in s


class TestGNARValidation:
    """Tests for input validation."""

    def test_bad_adjacency_not_square(self, ts_np, s_array):
        with pytest.raises(ValueError, match="square"):
            GNAR(np.array([[0, 1]]), p=2, s=s_array, ts=ts_np)

    def test_bad_adjacency_negative(self, ts_np, s_array):
        A = np.array([[0, -1, 0], [-1, 0, 1], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="non-negative"):
            GNAR(A, p=2, s=s_array, ts=ts_np)

    def test_bad_s_length(self, adjacency_3, ts_np):
        with pytest.raises(ValueError):
            GNAR(adjacency_3, p=2, s=np.array([1]), ts=ts_np)

    def test_bad_model_type(self, adjacency_3, ts_np, s_array):
        with pytest.raises(ValueError, match="Invalid model_type"):
            GNAR(adjacency_3, p=2, s=s_array, ts=ts_np, model_type="invalid")

    def test_no_ts_no_coeffs(self, adjacency_3, s_array):
        with pytest.raises(ValueError, match="Either"):
            GNAR(adjacency_3, p=2, s=s_array)
