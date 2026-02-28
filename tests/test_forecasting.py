import numpy as np
import pytest

from gnar import GNAR
from gnar.utils.data_utils import param_reorder


A_3NODE = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=float)


def _make_model(A, p, s, coeffs, model_type="standard", mean=0, sigma_2=1.0, net_type="unweighted"):
    return GNAR(A, p=p, s=s, coeffs=coeffs, model_type=model_type,
                mean=mean, sigma_2=sigma_2, net_type=net_type)


class TestOneStepForecast:

    def test_p1s1_hand_computed(self):
        # Path graph 0--1--2
        # ns_mats[0] = [[0, 0.5, 0], [1, 0, 1], [0, 0.5, 0]]
        # coeffs (p+sum(s), d) = (2, 3): row 0 = alpha_1, row 1 = beta_11
        # standard model: alphas differ, betas identical
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=0)
        last_obs = np.array([[1.0, 2.0, 3.0]])
        pred = G.predict(ts=last_obs, h=1)
        # pred shape: (h, d) = (1, 3)
        # ns_sum[i] = sum_j ts[j] * ns[0,j,i]
        # Node 0: ns_sum = 1*0+2*1+3*0=2, pred = 0.5*1 + 0.1*2 = 0.7
        # Node 1: ns_sum = 1*0.5+2*0+3*0.5=2, pred = 0.3*2 + 0.1*2 = 0.8
        # Node 2: ns_sum = 1*0+2*1+3*0=2, pred = 0.4*3 + 0.1*2 = 1.4
        expected = np.array([0.7, 0.8, 1.4])
        np.testing.assert_allclose(pred[0], expected, atol=1e-10)

    def test_p2s11_hand_computed(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.2, 0.1, 0.15],
                           [0.1, 0.1, 0.1],
                           [0.05, 0.05, 0.05]])
        G = _make_model(A_3NODE, p=2, s=np.array([1, 1]), coeffs=coeffs, mean=0)
        # Need 2 observations for p=2
        ts = np.array([[1.0, 2.0, 3.0],
                       [0.5, 1.0, 1.5]])
        pred = G.predict(ts=ts, h=1)
        # Design matrix: [lag1_ts, lag2_ts, lag1_ns, lag2_ns]
        # lag1 = ts[1] = [0.5, 1.0, 1.5], lag2 = ts[0] = [1.0, 2.0, 3.0]
        # ns_mats[0] for lag1: ns_sum = ns_mats[0].T @ [0.5, 1.0, 1.5]
        # Node 0: 0.5*1.0 = 0.5, Node 1: 0.5*0.5 + 0.5*1.5 = 1.0, Node 2: 0.5*1.0 = 0.5
        # Wait, ns_sum for node i = sum_j ts[j] * ns_mats[0, j, i]
        # Node 0: ts[0]*ns[0,0,0] + ts[1]*ns[0,1,0] + ts[2]*ns[0,2,0] = 0.5*0 + 1.0*1 + 1.5*0 = 1.0
        # Node 1: 0.5*0.5 + 1.0*0 + 1.5*0.5 = 1.0
        # Node 2: 0.5*0 + 1.0*1 + 1.5*0 = 1.0
        # ns for lag2: ts[0] = [1.0, 2.0, 3.0]
        # Node 0: 1*0 + 2*1 + 3*0 = 2.0
        # Node 1: 1*0.5 + 2*0 + 3*0.5 = 2.0
        # Node 2: 1*0 + 2*1 + 3*0 = 2.0
        # X = [alpha1*lag1, alpha2*lag2, beta11*ns_lag1, beta21*ns_lag2]
        # Node 0: 0.5*0.5 + 0.2*1.0 + 0.1*1.0 + 0.05*2.0 = 0.25+0.2+0.1+0.1 = 0.65
        # Node 1: 0.3*1.0 + 0.1*2.0 + 0.1*1.0 + 0.05*2.0 = 0.3+0.1+0.1+0.1 = 0.6  -- wait 0.1*2 = 0.2
        # Node 1: 0.3*1.0 + 0.1*2.0 + 0.1*1.0 + 0.05*2.0 = 0.3+0.2+0.1+0.1 = 0.7
        # Node 2: 0.4*1.5 + 0.15*3.0 + 0.1*1.0 + 0.05*2.0 = 0.6+0.45+0.1+0.1 = 1.25
        expected = np.array([0.65, 0.7, 1.25])
        np.testing.assert_allclose(pred[0], expected, atol=1e-10)

    def test_with_nonzero_mean(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        mean = np.array([1.0, 2.0, 3.0])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=mean)
        last_obs = np.array([[2.0, 4.0, 6.0]])  # = mean + [1, 2, 3]
        pred = G.predict(ts=last_obs, h=1)
        # Demeaned obs: [1, 2, 3], same as test_p1s1_hand_computed
        # Prediction (demeaned): [0.7, 0.8, 1.4]
        # Add mean back: [1.7, 2.8, 4.4]
        expected = np.array([1.7, 2.8, 4.4])
        np.testing.assert_allclose(pred[0], expected, atol=1e-10)


class TestMultiStepForecast:

    def test_2step_hand_computed(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=0)
        last_obs = np.array([[1.0, 2.0, 3.0]])
        pred = G.predict(ts=last_obs, h=2)
        # h1 = [0.7, 0.8, 1.4] (from TestOneStepForecast)
        # h2: ns_sum[i] = sum_j h1[j] * ns[0,j,i]
        # Node 0: 0.7*0 + 0.8*1 + 1.4*0 = 0.8, pred = 0.5*0.7 + 0.1*0.8 = 0.43
        # Node 1: 0.7*0.5 + 0.8*0 + 1.4*0.5 = 1.05, pred = 0.3*0.8 + 0.1*1.05 = 0.345
        # Node 2: 0.7*0 + 0.8*1 + 1.4*0 = 0.8, pred = 0.4*1.4 + 0.1*0.8 = 0.64
        expected_h2 = np.array([0.43, 0.345, 0.64])
        np.testing.assert_allclose(pred[1], expected_h2, atol=1e-10)

    def test_batch_predictions_shape(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=0)
        ts = np.random.normal(0, 1, (10, 3))
        preds = G.predict(ts=ts, h=3)
        # Shape: (n - p + 1, d, h) = (10, 3, 3)
        assert preds.shape == (10, 3, 3)


class TestGNARvsVARConsistency:

    def _check_consistency(self, G, ts, h):
        V = G.to_var()
        gnar_pred = G.predict(ts=ts, h=h)
        var_pred = V.predict(ts=ts, h=h)
        np.testing.assert_allclose(gnar_pred, var_pred, atol=1e-10)

    def test_1step_standard(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, model_type="standard")
        self._check_consistency(G, ts[-1:], h=1)

    def test_5step_standard(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, model_type="standard")
        self._check_consistency(G, ts[-1:], h=5)

    def test_global_consistency(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, model_type="global")
        self._check_consistency(G, ts[-1:], h=3)

    def test_local_consistency(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, model_type="local")
        self._check_consistency(G, ts[-1:], h=3)

    def test_p2_consistency(self):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(A_3NODE, p=2, s=np.array([1, 1]), ts=ts, model_type="standard")
        self._check_consistency(G, ts[-2:], h=3)

    def test_weighted_consistency(self, weighted_path_3):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(weighted_path_3, p=1, s=np.array([1]), ts=ts, model_type="standard", net_type="weighted")
        self._check_consistency(G, ts[-1:], h=3)

    def test_distance_consistency(self, distance_path_3):
        np.random.seed(42)
        ts = np.random.normal(0, 1, (50, 3))
        G = GNAR(distance_path_3, p=1, s=np.array([1]), ts=ts, model_type="standard", net_type="distance")
        self._check_consistency(G, ts[-1:], h=3)


class TestGNARToVARConversion:

    def test_var_coeffs_exact_p1s1(self):
        # GNAR coeffs: alpha=[0.5, 0.3, 0.4], beta=[0.1, 0.1, 0.1]
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=0)
        V = G.to_var()
        # VAR coeffs: Phi[j, i] = alpha_i * delta(i,j) + beta * ns[0, j, i]
        # ns[0] = [[0, 0.5, 0], [1, 0, 1], [0, 0.5, 0]]
        # Column 0 (node 0): Phi[:, 0] = [0.5, 0, 0] + 0.1*[0, 1, 0] = [0.5, 0.1, 0]
        # Column 1 (node 1): Phi[:, 1] = [0, 0.3, 0] + 0.1*[0.5, 0, 0.5] = [0.05, 0.3, 0.05]
        # Column 2 (node 2): Phi[:, 2] = [0, 0, 0.4] + 0.1*[0, 1, 0] = [0, 0.1, 0.4]
        expected = np.array([[0.5,  0.05, 0.0],
                             [0.1,  0.3,  0.1],
                             [0.0,  0.05, 0.4]])
        np.testing.assert_allclose(V.coeffs, expected, atol=1e-10)

    def test_var_coeffs_exact_p2s11(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.2, 0.1, 0.15],
                           [0.1, 0.1, 0.1],
                           [0.05, 0.05, 0.05]])
        G = _make_model(A_3NODE, p=2, s=np.array([1, 1]), coeffs=coeffs, mean=0)
        V = G.to_var()
        assert V.coeffs.shape == (6, 3)
        # Lag-1 block (rows 0-2): same logic as p1s1 with alpha1, beta11
        expected_lag1 = np.array([[0.5,  0.05, 0.0],
                                  [0.1,  0.3,  0.1],
                                  [0.0,  0.05, 0.4]])
        np.testing.assert_allclose(V.coeffs[:3], expected_lag1, atol=1e-10)
        # Lag-2 block (rows 3-5): alpha2, beta21
        expected_lag2 = np.array([[0.2,   0.025, 0.0],
                                  [0.05,  0.1,   0.05],
                                  [0.0,   0.025, 0.15]])
        np.testing.assert_allclose(V.coeffs[3:], expected_lag2, atol=1e-10)

    def test_preserves_mean_and_sigma2(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        mean = np.array([1.0, 2.0, 3.0])
        sigma_2 = np.eye(3) * 0.5
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=mean, sigma_2=sigma_2)
        V = G.to_var()
        np.testing.assert_allclose(V.mu, G.mu, atol=1e-10)
        np.testing.assert_allclose(V.sigma_2, G.sigma_2, atol=1e-10)


class TestParamReorder:

    def test_p1s1_no_change(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        reordered = param_reorder(coeffs, p=1, s=np.array([1]))
        np.testing.assert_allclose(reordered, coeffs, atol=1e-10)

    def test_p2s11_interleaves(self):
        # Original order: [a1, a2, b11, b21]
        coeffs = np.array([[0.5, 0.3, 0.4],    # a1
                           [0.2, 0.1, 0.15],   # a2
                           [0.1, 0.1, 0.1],    # b11
                           [0.05, 0.05, 0.05]]) # b21
        reordered = param_reorder(coeffs, p=2, s=np.array([1, 1]))
        # Expected: [a1, b11, a2, b21]
        expected = np.array([[0.5, 0.3, 0.4],
                             [0.1, 0.1, 0.1],
                             [0.2, 0.1, 0.15],
                             [0.05, 0.05, 0.05]])
        np.testing.assert_allclose(reordered, expected, atol=1e-10)


class TestForecastValidation:

    def test_no_ts_raises(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=0)
        with pytest.raises(ValueError, match="not fit to data"):
            G.predict()

    def test_wrong_d_raises(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.1, 0.1]])
        G = _make_model(A_3NODE, p=1, s=np.array([1]), coeffs=coeffs, mean=0)
        with pytest.raises(ValueError):
            G.predict(ts=np.ones((1, 5)))

    def test_too_few_obs_raises(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.2, 0.1, 0.15],
                           [0.1, 0.1, 0.1],
                           [0.05, 0.05, 0.05]])
        G = _make_model(A_3NODE, p=2, s=np.array([1, 1]), coeffs=coeffs, mean=0)
        with pytest.raises(ValueError, match="insufficient"):
            G.predict(ts=np.ones((1, 3)))
