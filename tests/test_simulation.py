import numpy as np
import pytest

from gnar import GNAR


A_3NODE = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=float)


def _stationary_model(mean=0):
    coeffs = np.array([[0.3, 0.2, 0.4],
                       [0.1, 0.1, 0.1]])
    return GNAR(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=mean, sigma_2=1.0)


class TestSimulationMean:

    def test_zero_mean_model(self):
        np.random.seed(42)
        G = _stationary_model(mean=0)
        sim = G.simulate(10000, burn_in=200)
        np.testing.assert_allclose(np.mean(sim, axis=0), np.zeros(3), atol=0.15)

    def test_nonzero_mean_model(self):
        np.random.seed(42)
        mean = np.array([1.0, 2.0, 3.0])
        G = _stationary_model(mean=mean)
        sim = G.simulate(10000, burn_in=200)
        np.testing.assert_allclose(np.mean(sim, axis=0), mean, atol=0.15)


class TestSimulationAutocovariance:

    def test_lag0_matches_theoretical(self):
        np.random.seed(42)
        G = _stationary_model(mean=0)
        sim = G.simulate(50000, burn_in=500)
        sim = sim - np.mean(sim, axis=0)
        sample_gamma0 = sim.T @ sim / len(sim)
        theoretical = G.compute_autocov_mats(max_lag=0)
        np.testing.assert_allclose(sample_gamma0, theoretical[0], atol=0.15)

    def test_lag1_matches_theoretical(self):
        np.random.seed(42)
        G = _stationary_model(mean=0)
        sim = G.simulate(50000, burn_in=500)
        sim = sim - np.mean(sim, axis=0)
        n = len(sim)
        sample_gamma1 = sim[1:].T @ sim[:n-1] / (n - 1)
        theoretical = G.compute_autocov_mats(max_lag=1)
        np.testing.assert_allclose(sample_gamma1, theoretical[1], atol=0.15)

    def test_multiple_lags(self):
        np.random.seed(42)
        G = _stationary_model(mean=0)
        sim = G.simulate(50000, burn_in=500)
        sim = sim - np.mean(sim, axis=0)
        n = len(sim)
        theoretical = G.compute_autocov_mats(max_lag=3)
        for lag in range(4):
            if lag == 0:
                sample = sim.T @ sim / n
            else:
                sample = sim[lag:].T @ sim[:n-lag] / (n - lag)
            np.testing.assert_allclose(sample, theoretical[lag], atol=0.15)


class TestSimulationReproducibility:

    def test_same_seed_same_output(self):
        G = _stationary_model(mean=0)
        np.random.seed(123)
        sim1 = G.simulate(100)
        np.random.seed(123)
        sim2 = G.simulate(100)
        np.testing.assert_allclose(sim1, sim2, atol=1e-10)

    def test_different_seed_different_output(self):
        G = _stationary_model(mean=0)
        np.random.seed(123)
        sim1 = G.simulate(100)
        np.random.seed(456)
        sim2 = G.simulate(100)
        assert not np.allclose(sim1, sim2)


class TestSimulationShape:

    def test_output_shape(self):
        G = _stationary_model(mean=0)
        np.random.seed(42)
        sim = G.simulate(200)
        assert sim.shape == (200, 3)
