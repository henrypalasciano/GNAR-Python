import numpy as np
import pandas as pd
import pytest
import warnings

from gnar import VAR


class TestVARFitting:

    def test_recover_coefficients_p1(self):
        np.random.seed(42)
        true_coeffs = np.array([[0.5, 0.1],
                                [0.2, 0.3]])
        V_true = VAR(p=1, coeffs=true_coeffs, mean=0, sigma_2=1.0)
        ts = V_true.simulate(5000, burn_in=200)
        V_fit = VAR(p=1, ts=ts)
        np.testing.assert_allclose(V_fit.coeffs, true_coeffs, atol=0.05)

    def test_recover_coefficients_p2(self):
        np.random.seed(42)
        true_coeffs = np.array([[0.3, 0.1],
                                [0.1, 0.2],
                                [0.1, 0.05],
                                [0.05, 0.1]])
        V_true = VAR(p=2, coeffs=true_coeffs, mean=0, sigma_2=1.0)
        ts = V_true.simulate(5000, burn_in=200)
        V_fit = VAR(p=2, ts=ts)
        np.testing.assert_allclose(V_fit.coeffs, true_coeffs, atol=0.05)

    def test_sigma2_recovery(self):
        np.random.seed(42)
        true_coeffs = np.array([[0.3, 0.1],
                                [0.1, 0.2]])
        true_sigma = np.array([[1.0, 0.3],
                               [0.3, 1.0]])
        V_true = VAR(p=1, coeffs=true_coeffs, mean=0, sigma_2=true_sigma)
        ts = V_true.simulate(10000, burn_in=200)
        V_fit = VAR(p=1, ts=ts)
        np.testing.assert_allclose(V_fit.sigma_2, true_sigma, atol=0.1)

    def test_dataframe_input(self):
        np.random.seed(42)
        ts = pd.DataFrame(np.random.normal(0, 1, (100, 3)), columns=["A", "B", "C"])
        V = VAR(p=2, ts=ts)
        assert V.coeffs.shape == (6, 3)


class TestVARPredict:

    def test_1step_hand_computed(self):
        Phi = np.array([[0.5, 0.1],
                        [0.2, 0.3]])
        V = VAR(p=1, coeffs=Phi, mean=0, sigma_2=1.0)
        x = np.array([[1.0, 2.0]])
        pred = V.predict(ts=x, h=1)
        # pred = Phi.T @ x = [0.5*1+0.2*2, 0.1*1+0.3*2] = [0.9, 0.7]
        # Wait: VAR.predict does X @ coeffs where X = x (for p=1)
        # coeffs shape (p*d, d) = (2, 2), X shape (1, 2)
        # pred = X @ coeffs = [1, 2] @ [[0.5, 0.1], [0.2, 0.3]] = [0.9, 0.7]
        expected = np.array([0.9, 0.7])
        np.testing.assert_allclose(pred[0], expected, atol=1e-10)

    def test_2step_hand_computed(self):
        Phi = np.array([[0.5, 0.1],
                        [0.2, 0.3]])
        V = VAR(p=1, coeffs=Phi, mean=0, sigma_2=1.0)
        x = np.array([[1.0, 2.0]])
        pred = V.predict(ts=x, h=2)
        h1 = np.array([0.9, 0.7])
        # h2 = h1 @ Phi = [0.9*0.5+0.7*0.2, 0.9*0.1+0.7*0.3] = [0.59, 0.30]
        expected_h2 = np.array([0.59, 0.30])
        np.testing.assert_allclose(pred[1], expected_h2, atol=1e-10)

    def test_batch_shape(self):
        Phi = np.array([[0.5, 0.1],
                        [0.2, 0.3]])
        V = VAR(p=1, coeffs=Phi, mean=0, sigma_2=1.0)
        ts = np.random.normal(0, 1, (10, 2))
        preds = V.predict(ts=ts, h=3)
        assert preds.shape == (10, 2, 3)


class TestVARStationarity:

    def test_stationary_model(self):
        coeffs = 0.3 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        assert V.is_stationary()

    def test_nonstationary_model(self):
        coeffs = 2.0 * np.eye(3)
        with pytest.warns(UserWarning, match="non-stationary"):
            V = VAR(p=1, coeffs=coeffs)
        assert not V.is_stationary()

    def test_companion_form_p1(self):
        coeffs = np.array([[0.5, 0.1],
                           [0.2, 0.3]])
        V = VAR(p=1, coeffs=coeffs, mean=0, sigma_2=1.0)
        phi, sigma = V.companion_form()
        np.testing.assert_allclose(phi, coeffs.T, atol=1e-10)

    def test_companion_form_p2_structure(self):
        d = 2
        coeffs = np.array([[0.3, 0.1],
                           [0.1, 0.2],
                           [0.1, 0.05],
                           [0.05, 0.1]])
        V = VAR(p=2, coeffs=coeffs, mean=0, sigma_2=1.0)
        phi, sigma = V.companion_form()
        assert phi.shape == (4, 4)
        # Top block: [Phi1.T, Phi2.T] transposed -> phi[0:2, :] = coeffs.T
        np.testing.assert_allclose(phi[:d], coeffs.T, atol=1e-10)
        # Bottom block: [I, 0]
        np.testing.assert_allclose(phi[d:, :d], np.eye(d), atol=1e-10)
        np.testing.assert_allclose(phi[d:, d:], np.zeros((d, d)), atol=1e-10)


class TestVARAutocovariance:

    def test_gamma0_symmetric(self):
        np.random.seed(42)
        coeffs = np.array([[0.5, 0.1],
                           [0.2, 0.3]])
        V = VAR(p=1, coeffs=coeffs, mean=0, sigma_2=1.0)
        gamma = V.compute_autocov_mats(max_lag=0)
        np.testing.assert_allclose(gamma[0], gamma[0].T, atol=1e-10)

    def test_gamma0_positive_definite(self):
        coeffs = np.array([[0.5, 0.1],
                           [0.2, 0.3]])
        V = VAR(p=1, coeffs=coeffs, mean=0, sigma_2=1.0)
        gamma = V.compute_autocov_mats(max_lag=0)
        eigvals = np.linalg.eigvalsh(gamma[0])
        assert np.all(eigvals > 0)

    def test_autocorr_lag0_diagonal_ones(self):
        coeffs = np.array([[0.5, 0.1],
                           [0.2, 0.3]])
        V = VAR(p=1, coeffs=coeffs, mean=0, sigma_2=1.0)
        R = V.compute_autocorr_mats(max_lag=0)
        np.testing.assert_allclose(np.diag(R[0]), np.ones(2), atol=1e-10)

    def test_yule_walker_recursion(self):
        coeffs = np.array([[0.3, 0.1],
                           [0.1, 0.2],
                           [0.1, 0.05],
                           [0.05, 0.1]])
        V = VAR(p=2, coeffs=coeffs, mean=0, sigma_2=1.0)
        gamma = V.compute_autocov_mats(max_lag=5)
        Phi1 = coeffs[:2].T
        Phi2 = coeffs[2:].T
        for h in range(2, 6):
            expected = Phi1 @ gamma[h-1] + Phi2 @ gamma[h-2]
            np.testing.assert_allclose(gamma[h], expected, atol=1e-10)

    def test_autocov_vs_simulation(self):
        np.random.seed(42)
        coeffs = np.array([[0.5, 0.1],
                           [0.2, 0.3]])
        V = VAR(p=1, coeffs=coeffs, mean=0, sigma_2=1.0)
        sim = V.simulate(50000, burn_in=500)
        sim = sim - np.mean(sim, axis=0)
        n = len(sim)
        theoretical = V.compute_autocov_mats(max_lag=2)
        for lag in range(3):
            if lag == 0:
                sample = sim.T @ sim / n
            else:
                sample = sim[lag:].T @ sim[:n-lag] / (n - lag)
            np.testing.assert_allclose(sample, theoretical[lag], atol=0.15)


class TestVARInfoCriteria:

    def test_bic_not_fitted_raises(self):
        coeffs = 0.1 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit"):
            V.bic()

    def test_aic_not_fitted_raises(self):
        coeffs = 0.1 * np.eye(3)
        V = VAR(p=1, coeffs=coeffs)
        with pytest.raises(ValueError, match="not fit"):
            V.aic()

    def test_bic_favours_true_model(self):
        np.random.seed(42)
        true_coeffs = np.array([[0.5, 0.1],
                                [0.2, 0.3]])
        V_true = VAR(p=1, coeffs=true_coeffs, mean=0, sigma_2=1.0)
        ts = V_true.simulate(1000, burn_in=200)
        V1 = VAR(p=1, ts=ts)
        V3 = VAR(p=3, ts=ts)
        assert V1.bic() < V3.bic()
