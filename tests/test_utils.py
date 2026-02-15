import numpy as np
import pytest

from gnar.utils.neighbour_sets import neighbour_set_mats, compute_neighbour_sums
from gnar.utils.data_utils import gnar_checks, set_mean, set_cov, cov_mat, check_gnar_coeffs
from gnar.utils.simulating import generate_noise


class TestNeighbourSets:
    """Tests for neighbour set computation."""

    def test_ns_mats_shape(self, adjacency_3):
        ns = neighbour_set_mats(adjacency_3, r=2)
        assert ns.shape == (2, 3, 3)

    def test_ns_mats_normalization(self, adjacency_3):
        ns = neighbour_set_mats(adjacency_3, r=1)
        # Each column should sum to 1 (or 0 if no neighbours)
        col_sums = np.sum(ns[0], axis=0)
        for s in col_sums:
            assert s == pytest.approx(0.0) or s == pytest.approx(1.0)

    def test_ns_mats_stage2(self, adjacency_5):
        ns = neighbour_set_mats(adjacency_5, r=2)
        assert ns.shape == (2, 5, 5)
        # Stage 2 should have nonzero entries (2-hop neighbours exist in cycle)
        assert np.any(ns[1] > 0)

    def test_compute_neighbour_sums_shape(self, adjacency_3):
        ns = neighbour_set_mats(adjacency_3, r=2)
        ts = np.random.normal(0, 1, (50, 3))
        data = compute_neighbour_sums(ts, ns, r=2)
        assert data.shape == (50, 3, 3)
        # First slice should be the original time series
        assert np.allclose(data[:, :, 0], ts)


class TestGNARChecks:
    """Tests for input validation functions."""

    def test_valid_inputs(self, adjacency_3, s_array):
        assert gnar_checks(adjacency_3, 2, s_array, "standard") is None

    def test_non_square_adjacency(self, s_array):
        with pytest.raises(ValueError, match="square"):
            gnar_checks(np.array([[0, 1]]), 2, s_array, "standard")

    def test_negative_adjacency(self, s_array):
        A = np.array([[0, -1], [-1, 0]], dtype=float)
        with pytest.raises(ValueError, match="non-negative"):
            gnar_checks(A, 2, s_array, "standard")

    def test_bad_p(self, adjacency_3, s_array):
        with pytest.raises(ValueError, match="lags"):
            gnar_checks(adjacency_3, 0, s_array, "standard")

    def test_bad_s_length(self, adjacency_3):
        with pytest.raises(ValueError):
            gnar_checks(adjacency_3, 2, np.array([1, 1, 1]), "standard")

    def test_bad_model_type(self, adjacency_3, s_array):
        with pytest.raises(ValueError, match="Invalid model_type"):
            gnar_checks(adjacency_3, 2, s_array, "bad")


class TestCheckGNARCoeffs:
    """Tests for coefficient validation."""

    def test_valid_global_coeffs(self):
        # Global: all columns identical
        coeffs = np.array([[0.5, 0.5, 0.5],
                           [0.1, 0.1, 0.1],
                           [0.2, 0.2, 0.2]])
        assert check_gnar_coeffs(coeffs, d=3, p=2, s=np.array([1]), model_type="global") is None

    def test_valid_standard_coeffs(self):
        # Standard: beta rows identical, alpha rows can differ
        coeffs = np.array([[0.5, 0.3, 0.4],
                           [0.1, 0.2, 0.15],
                           [0.2, 0.2, 0.2]])
        assert check_gnar_coeffs(coeffs, d=3, p=2, s=np.array([1]), model_type="standard") is None

    def test_invalid_global_coeffs(self):
        coeffs = np.array([[0.5, 0.3, 0.5],
                           [0.1, 0.1, 0.1]])
        with pytest.raises(ValueError, match="global"):
            check_gnar_coeffs(coeffs, d=3, p=1, s=np.array([1]), model_type="global")

    def test_wrong_shape(self):
        coeffs = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match="number of coefficients"):
            check_gnar_coeffs(coeffs, d=3, p=2, s=np.array([1]), model_type="local")


class TestSetMean:
    """Tests for set_mean function."""

    def test_scalar(self):
        mu = set_mean(5.0, 3)
        assert mu.shape == (1, 3)
        assert np.all(mu == 5.0)

    def test_array(self):
        mu = set_mean(np.array([1, 2, 3]), 3)
        assert mu.shape == (1, 3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            set_mean("bad", 3)


class TestSetCov:
    """Tests for set_cov function."""

    def test_scalar(self):
        assert set_cov(1.5) == 1.5

    def test_array(self):
        arr = np.eye(3)
        result = set_cov(arr)
        assert np.array_equal(result, arr)

    def test_invalid(self):
        with pytest.raises(ValueError):
            set_cov("bad")


class TestCovMat:
    """Tests for cov_mat function."""

    def test_scalar(self):
        result = cov_mat(2.0, 3)
        expected = 2.0 * np.eye(3)
        assert np.allclose(result, expected)

    def test_vector(self):
        result = cov_mat(np.array([1.0, 2.0, 3.0]), 3)
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result), [1.0, 2.0, 3.0])

    def test_matrix(self):
        mat = np.eye(3) * 5
        result = cov_mat(mat, 3)
        assert np.allclose(result, mat)


class TestGenerateNoise:
    """Tests for noise generation."""

    def test_scalar_sigma(self):
        noise = generate_noise(1.0, 100, 3)
        assert noise.shape == (100, 3)

    def test_vector_sigma(self):
        noise = generate_noise(np.array([1.0, 2.0, 3.0]), 100, 3)
        assert noise.shape == (100, 3)

    def test_matrix_sigma(self):
        noise = generate_noise(np.eye(3), 100, 3)
        assert noise.shape == (100, 3)
