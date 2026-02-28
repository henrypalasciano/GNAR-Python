import numpy as np
import pytest

from gnar.utils.neighbour_sets import neighbour_set_mats, compute_neighbour_sums, weight_mats


class TestUnweighted:

    def test_path3_stage1_exact(self, path_graph_3):
        ns = neighbour_set_mats(path_graph_3, r=1)
        expected = np.array([[0, 0.5, 0],
                             [1, 0,   1],
                             [0, 0.5, 0]])
        np.testing.assert_allclose(ns[0], expected, atol=1e-10)

    def test_path3_stage2_exact(self, path_graph_3):
        ns = neighbour_set_mats(path_graph_3, r=2)
        expected = np.array([[0, 0, 1],
                             [0, 0, 0],
                             [1, 0, 0]])
        np.testing.assert_allclose(ns[1], expected, atol=1e-10)

    def test_complete3_stage1_exact(self, complete_graph_3):
        ns = neighbour_set_mats(complete_graph_3, r=1)
        expected = 0.5 * (np.ones((3, 3)) - np.eye(3))
        np.testing.assert_allclose(ns[0], expected, atol=1e-10)

    def test_complete3_stage2_all_zero(self, complete_graph_3):
        ns = neighbour_set_mats(complete_graph_3, r=2)
        np.testing.assert_allclose(ns[1], np.zeros((3, 3)), atol=1e-10)

    def test_cycle5_stage1_exact(self, cycle_graph_5):
        ns = neighbour_set_mats(cycle_graph_5, r=1)
        for j in range(5):
            col = ns[0, :, j]
            assert col[j] == pytest.approx(0.0)
            neighbours = [(j - 1) % 5, (j + 1) % 5]
            for k in neighbours:
                assert col[k] == pytest.approx(0.5)
            non_neighbours = [k for k in range(5) if k != j and k not in neighbours]
            for k in non_neighbours:
                assert col[k] == pytest.approx(0.0)

    def test_cycle5_stage2_exact(self, cycle_graph_5):
        ns = neighbour_set_mats(cycle_graph_5, r=2)
        for j in range(5):
            col = ns[1, :, j]
            stage2 = [(j - 2) % 5, (j + 2) % 5]
            for k in stage2:
                assert col[k] == pytest.approx(0.5)
            others = [k for k in range(5) if k not in stage2]
            for k in others:
                assert col[k] == pytest.approx(0.0)

    def test_column_normalisation(self, path_graph_3, cycle_graph_5, complete_graph_3, diamond_graph_4):
        for A in [path_graph_3, cycle_graph_5, complete_graph_3, diamond_graph_4]:
            ns = neighbour_set_mats(A, r=2)
            for stage in range(2):
                col_sums = np.sum(ns[stage], axis=0)
                for s in col_sums:
                    assert s == pytest.approx(0.0) or s == pytest.approx(1.0)

    def test_stage_isolation(self, cycle_graph_5):
        ns = neighbour_set_mats(cycle_graph_5, r=2)
        nz0 = set(zip(*np.nonzero(ns[0])))
        nz1 = set(zip(*np.nonzero(ns[1])))
        assert nz0.isdisjoint(nz1)
        for i, j in nz0 | nz1:
            assert i != j


class TestWeighted:

    def test_weighted_path3_stage1(self, weighted_path_3):
        ns = neighbour_set_mats(weighted_path_3, r=1, net_type="weighted")
        # Node 1 (col 1) neighbours: node 0 (weight 2), node 2 (weight 3)
        assert ns[0, 0, 1] == pytest.approx(2 / 5)
        assert ns[0, 2, 1] == pytest.approx(3 / 5)

    def test_weighted_diamond_stage2(self, weighted_diamond_4):
        ns = neighbour_set_mats(weighted_diamond_4, r=2, net_type="weighted")
        # Node 0 and node 3 are stage-2 neighbours
        # (3,0): paths 3->1->0 (5*2=10) and 3->2->0 (7*3=21), total=31
        # Only stage-2 neighbour for node 3 is node 0, so normalised = 1
        assert ns[1, 0, 3] == pytest.approx(1.0)
        assert ns[1, 3, 0] == pytest.approx(1.0)

    def test_weighted_binary_matches_unweighted(self, path_graph_3):
        ns_uw = neighbour_set_mats(path_graph_3, r=2, net_type="unweighted")
        ns_w = neighbour_set_mats(path_graph_3, r=2, net_type="weighted")
        np.testing.assert_allclose(ns_uw, ns_w, atol=1e-10)


class TestDistance:

    def test_distance_path3_stage1(self, distance_path_3):
        ns = neighbour_set_mats(distance_path_3, r=1, net_type="distance")
        # Node 1 (col 1): distances 4, 2 → weights 1/4, 1/2 → normalised 1/3, 2/3
        assert ns[0, 0, 1] == pytest.approx(1 / 3)
        assert ns[0, 2, 1] == pytest.approx(2 / 3)


class TestComputeNeighbourSums:

    def test_first_slice_is_ts(self, path_graph_3):
        ns = neighbour_set_mats(path_graph_3, r=1)
        ts = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        data = compute_neighbour_sums(ts, ns, r=1)
        np.testing.assert_allclose(data[:, :, 0], ts, atol=1e-10)

    def test_stage1_sum_exact(self, path_graph_3):
        ns = neighbour_set_mats(path_graph_3, r=1)
        ts = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        data = compute_neighbour_sums(ts, ns, r=1)
        # Stage-1 neighbour sums for t=0: node 0 gets 0.5*2+0*3=1, node 1 gets 1*1+1*3=4, node 2 gets 0.5*2+0*1=1
        # Actually: ns[0] @ ts[0] gives the neighbour sums
        expected_t0 = ns[0].T @ ts[0]
        # Wait - compute_neighbour_sums does ts @ ns_mats
        # data[:,:,1:] = np.transpose(ts @ ns_mats, (1, 2, 0))
        # ts @ ns_mats has shape (2, 1, 3, 3) -> actually ts is (2,3), ns_mats is (1,3,3)
        # ts @ ns_mats -> broadcasting: (2, 3) @ (1, 3, 3) -> (2, 1, 3, 3)?
        # No: ts is (n, d), ns_mats is (r, d, d). ts @ ns_mats: (n, d) @ (r, d, d) -> numpy broadcasts to (r, n, d)
        # Actually (n, d) @ (r, d, d) -> each (d,) row of ts does (d,) @ (d, d) for each of r mats
        # Result shape: ts has shape (2, 3), ns_mats has (1, 3, 3).
        # np broadcasting: (2, 3) is treated as (1, 2, 3). (1, 3, 3). matmul: last two dims: (2,3) @ (3,3) -> (2,3), over batch dim 1. Result: (1, 2, 3)
        # Then transpose((1,2,0)) -> from (r, n, d) to (n, d, r)
        # So data[t, i, 1+s] = sum_j ts[t, j] * ns_mats[s, j, i]
        # For t=0, node 0, stage 1: sum_j ts[0,j] * ns[0,j,0] = 1*0 + 2*1 + 3*0 = 2
        expected = np.array([
            [2.0, 2.0, 2.0],   # t=0: node0=2*1=2, node1=1*0.5+3*0.5=2, node2=2*1=2
            [5.0, 5.0, 5.0],   # t=1: node0=5*1=5, node1=4*0.5+6*0.5=5, node2=5*1=5
        ])
        np.testing.assert_allclose(data[:, :, 1], expected, atol=1e-10)


class TestWeightMats:

    def test_p1s1_path3(self, path_graph_3):
        ns = neighbour_set_mats(path_graph_3, r=1)
        W = weight_mats(ns, p=1, s=np.array([1]), d=3)
        # W shape: (3, 3, 2) - for each node i, W[i] is (d, p+sum(s)) = (3, 2)
        assert W.shape == (3, 3, 2)
        # Column 0 should be identity column (alpha coeff)
        for i in range(3):
            assert W[i, i, 0] == pytest.approx(1.0)
        # Column 1 should be ns_mats column (beta coeff)
        for i in range(3):
            np.testing.assert_allclose(W[i, :, 1], ns[0, :, i], atol=1e-10)

    def test_p2s11_path3(self, path_graph_3):
        ns = neighbour_set_mats(path_graph_3, r=1)
        W = weight_mats(ns, p=2, s=np.array([1, 1]), d=3)
        # W shape: (3, 6, 4) - maps 4 GNAR params to 6 VAR params
        assert W.shape == (3, 6, 4)
