import numpy as np
import pytest

from gnar import GNAR


# Hardcoded time series for deterministic OLS tests
TS_2NODE = np.array([
    [ 0.1,  0.3],
    [-0.2,  0.5],
    [ 0.4, -0.1],
    [ 0.3,  0.2],
    [-0.1,  0.4],
    [ 0.2, -0.3],
    [ 0.5,  0.1],
    [-0.3,  0.6],
    [ 0.1, -0.2],
    [ 0.4,  0.3],
    [-0.2,  0.1],
    [ 0.3, -0.4],
    [ 0.1,  0.5],
    [-0.4,  0.2],
    [ 0.2,  0.1],
    [ 0.3, -0.1],
    [-0.1,  0.3],
    [ 0.4, -0.2],
    [ 0.2,  0.4],
    [-0.3,  0.1],
])

TS_3NODE = np.array([
    [ 0.5, -0.2,  0.3],
    [-0.1,  0.4,  0.2],
    [ 0.3, -0.3,  0.1],
    [ 0.2,  0.1, -0.2],
    [-0.4,  0.5,  0.3],
    [ 0.1, -0.1,  0.4],
    [ 0.3,  0.2, -0.3],
    [-0.2,  0.3,  0.1],
    [ 0.4, -0.4,  0.2],
    [ 0.1,  0.1, -0.1],
    [-0.3,  0.4,  0.3],
    [ 0.2, -0.2,  0.1],
    [ 0.3,  0.1, -0.4],
    [-0.1,  0.3,  0.2],
    [ 0.4, -0.1,  0.1],
    [ 0.2,  0.2, -0.2],
    [-0.3,  0.4,  0.3],
    [ 0.1, -0.3,  0.2],
    [ 0.3,  0.1, -0.1],
    [-0.2,  0.2,  0.4],
])

A_2NODE = np.array([[0, 1], [1, 0]], dtype=float)

A_3NODE = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=float)


class TestOLSFitting:

    def test_standard_p1s1_2node(self):
        G = GNAR(A_2NODE, p=1, s=np.array([1]), ts=TS_2NODE, model_type="standard", method="OLS")
        # Verify coefficients are reproducible and have correct shape
        assert G.coeffs.shape == (2, 2)
        # Re-fit to confirm determinism
        G2 = GNAR(A_2NODE, p=1, s=np.array([1]), ts=TS_2NODE, model_type="standard", method="OLS")
        np.testing.assert_allclose(G.coeffs, G2.coeffs, atol=1e-10)

    def test_standard_p1s1_3node(self):
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=TS_3NODE, model_type="standard", method="OLS")
        assert G.coeffs.shape == (2, 3)
        # Beta row (row 1) should be identical across all columns
        assert G.coeffs[1, 0] == pytest.approx(G.coeffs[1, 1], abs=1e-10)
        assert G.coeffs[1, 0] == pytest.approx(G.coeffs[1, 2], abs=1e-10)

    def test_global_p1s1(self):
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=TS_3NODE, model_type="global", method="OLS")
        assert G.coeffs.shape == (2, 3)
        # All columns identical
        np.testing.assert_allclose(G.coeffs[:, 0], G.coeffs[:, 1], atol=1e-10)
        np.testing.assert_allclose(G.coeffs[:, 0], G.coeffs[:, 2], atol=1e-10)

    def test_local_p1s1(self):
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=TS_3NODE, model_type="local", method="OLS")
        assert G.coeffs.shape == (2, 3)

    def test_standard_p2s11(self):
        G = GNAR(A_3NODE, p=2, s=np.array([1, 1]), ts=TS_3NODE, model_type="standard", method="OLS")
        assert G.coeffs.shape == (4, 3)
        # Beta rows (rows 2, 3) identical across columns
        np.testing.assert_allclose(G.coeffs[2, :], G.coeffs[2, 0] * np.ones(3), atol=1e-10)
        np.testing.assert_allclose(G.coeffs[3, :], G.coeffs[3, 0] * np.ones(3), atol=1e-10)

    def test_sigma2_correctness(self):
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=TS_3NODE, model_type="standard", method="OLS")
        assert G.sigma_2.shape == (3, 3)
        # sigma_2 should be symmetric
        np.testing.assert_allclose(G.sigma_2, G.sigma_2.T, atol=1e-10)
        # sigma_2 should be positive semi-definite
        eigvals = np.linalg.eigvalsh(G.sigma_2)
        assert np.all(eigvals >= -1e-10)


class TestYWFitting:

    def test_standard_p1s1(self):
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=TS_3NODE, model_type="standard", method="YW")
        assert G.coeffs.shape == (2, 3)
        # Beta row should be identical across columns
        assert G.coeffs[1, 0] == pytest.approx(G.coeffs[1, 1], abs=1e-10)

    def test_global_p1s1(self):
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=TS_3NODE, model_type="global", method="YW")
        # All columns identical
        np.testing.assert_allclose(G.coeffs[:, 0], G.coeffs[:, 1], atol=1e-10)


class TestOLSvsYW:

    def test_ols_yw_consistent_large_sample(self):
        np.random.seed(42)
        A = A_3NODE
        coeffs = np.array([[0.3, 0.2, 0.4],
                           [0.1, 0.1, 0.1]])
        G_true = GNAR(A, p=1, s=np.array([1]), coeffs=coeffs, mean=0, sigma_2=1.0)
        ts = G_true.simulate(5000, burn_in=200)
        G_ols = GNAR(A, p=1, s=np.array([1]), ts=ts, model_type="standard", method="OLS")
        G_yw = GNAR(A, p=1, s=np.array([1]), ts=ts, model_type="standard", method="YW")
        np.testing.assert_allclose(G_ols.coeffs, G_yw.coeffs, atol=0.05)


class TestWeightedFitting:

    def test_weighted_ols_fits(self, weighted_path_3):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (100, 3))
        G = GNAR(weighted_path_3, p=1, s=np.array([1]), ts=ts, net_type="weighted", method="OLS")
        assert G.coeffs.shape == (2, 3)

    def test_weighted_coefficient_recovery(self, weighted_path_3):
        np.random.seed(42)
        coeffs = np.array([[0.3, 0.2, 0.4],
                           [0.1, 0.1, 0.1]])
        G_true = GNAR(weighted_path_3, p=1, s=np.array([1]), coeffs=coeffs,
                       mean=0, sigma_2=1.0, net_type="weighted")
        ts = G_true.simulate(5000, burn_in=200)
        G_fit = GNAR(weighted_path_3, p=1, s=np.array([1]), ts=ts,
                      model_type="standard", net_type="weighted", method="OLS")
        np.testing.assert_allclose(G_fit.coeffs, coeffs, atol=0.05)

    def test_distance_ols_fits(self, distance_path_3):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (100, 3))
        G = GNAR(distance_path_3, p=1, s=np.array([1]), ts=ts, net_type="distance", method="OLS")
        assert G.coeffs.shape == (2, 3)


class TestDemean:

    def test_demean_true_recovers_mean(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (200, 3)) + np.array([1.0, 2.0, 3.0])
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, demean=True)
        np.testing.assert_allclose(G.mu.flatten(), np.mean(ts, axis=0), atol=1e-10)

    def test_demean_false_zero_mean(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (200, 3)) + np.array([1.0, 2.0, 3.0])
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, demean=False)
        np.testing.assert_allclose(G.mu.flatten(), np.zeros(3), atol=1e-10)
