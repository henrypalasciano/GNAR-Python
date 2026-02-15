import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def adjacency_3():
    """3-node path graph adjacency matrix."""
    return np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=float)


@pytest.fixture
def adjacency_5():
    """5-node cycle graph adjacency matrix."""
    A = np.zeros((5, 5))
    for i in range(5):
        A[i, (i + 1) % 5] = 1
        A[(i + 1) % 5, i] = 1
    return A


@pytest.fixture
def ts_np(adjacency_3):
    """Random time series as numpy array, shape (100, 3)."""
    np.random.seed(42)
    return np.random.normal(0, 1, (100, 3))


@pytest.fixture
def ts_df(ts_np):
    """Random time series as DataFrame."""
    return pd.DataFrame(ts_np, columns=["A", "B", "C"])


@pytest.fixture
def s_array():
    """Stage array for p=2."""
    return np.array([1, 1])
