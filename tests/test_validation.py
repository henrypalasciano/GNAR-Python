import numpy as np
import pytest

from gnar import GNAR, VAR
from gnar.utils.data_utils import check_gnar_coeffs


A_3NODE = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=float)


class TestGNARInputValidation:

    def test_non_square_adjacency(self):
        with pytest.raises(ValueError, match="square"):
            GNAR(np.array([[0, 1]]), p=1, s=np.array([1]),
                 coeffs=np.zeros((2, 2)))

    def test_negative_adjacency(self):
        A = np.array([[0, -1, 0], [-1, 0, 1], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="non-negative"):
            GNAR(A, p=1, s=np.array([1]), coeffs=np.zeros((2, 3)))

    def test_bad_p(self):
        with pytest.raises(ValueError, match="lags"):
            GNAR(A_3NODE, p=0, s=np.array([1]), coeffs=np.zeros((2, 3)))

    def test_bad_s_length(self):
        with pytest.raises(ValueError):
            GNAR(A_3NODE, p=2, s=np.array([1]), coeffs=np.zeros((3, 3)))

    def test_bad_model_type(self):
        with pytest.raises(ValueError, match="Invalid model_type"):
            GNAR(A_3NODE, p=1, s=np.array([1]), model_type="invalid",
                 coeffs=np.zeros((2, 3)))

    def test_bad_net_type(self):
        with pytest.raises(ValueError, match="Invalid net_type"):
            GNAR(A_3NODE, p=1, s=np.array([1]), net_type="invalid",
                 coeffs=np.zeros((2, 3)))

    def test_unweighted_rejects_weights(self):
        A = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        with pytest.raises(ValueError, match="binary"):
            GNAR(A, p=1, s=np.array([1]), net_type="unweighted",
                 coeffs=np.zeros((2, 3)))

    def test_weighted_accepts_weights(self):
        A = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
        coeffs = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        G = GNAR(A, p=1, s=np.array([1]), net_type="weighted",
                 coeffs=coeffs, model_type="global")
        assert G.coeffs.shape == (2, 3)

    def test_no_ts_no_coeffs(self):
        with pytest.raises(ValueError, match="Either"):
            GNAR(A_3NODE, p=1, s=np.array([1]))

    def test_bad_method(self):
        ts = np.random.normal(0, 1, (50, 3))
        with pytest.raises(ValueError, match="Method"):
            GNAR(A_3NODE, p=1, s=np.array([1]), ts=ts, method="BAD")


class TestCoeffValidation:

    def test_global_rejects_nonidentical(self):
        coeffs = np.array([[0.5, 0.3, 0.5],
                           [0.1, 0.1, 0.1]])
        with pytest.raises(ValueError, match="global"):
            check_gnar_coeffs(coeffs, d=3, p=1, s=np.array([1]), model_type="global")

    def test_standard_rejects_nonidentical_betas(self):
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.2, 0.1]])
        with pytest.raises(ValueError, match="standard"):
            check_gnar_coeffs(coeffs, d=3, p=1, s=np.array([1]), model_type="standard")

    def test_wrong_coeffs_shape(self):
        coeffs = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match="number of coefficients"):
            check_gnar_coeffs(coeffs, d=3, p=2, s=np.array([1]), model_type="local")


class TestVARInputValidation:

    def test_bad_p(self):
        with pytest.raises(ValueError, match="lags"):
            VAR(p=0, coeffs=np.eye(2))

    def test_no_ts_no_coeffs(self):
        with pytest.raises(ValueError, match="Either"):
            VAR(p=1)
