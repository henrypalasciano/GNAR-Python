import numpy as np

def neighbour_set_mats(A: np.ndarray, r: int, net_type: str = "unweighted") -> np.ndarray:
    """
    Compute a tensor containing the neighbour weight matrices up to stage r.

    For each stage, the weight matrix identifies nodes at that hop distance and
    assigns normalised weights. The normalisation depends on the network type:
      - "unweighted": uniform weights (1 / number of stage-r neighbours)
      - "weighted": weights proportional to connection strengths (products along paths)
      - "distance": weights inversely proportional to distances (products of 1/dist along paths)

    Params:
        A: np.array. Adjacency matrix. Shape (n, n). For unweighted networks, entries
            must be 0 or 1. For weighted/distance networks, entries are non-negative.
        r: int. Maximum stage of neighbour dependence.
        net_type: str. One of "unweighted", "weighted", or "distance".

    Returns:
        ns_mats: np.array. Tensor of neighbour weight matrices. Shape (r, n, n)
    """
    d = A.shape[0]
    ns_mats = np.zeros([r, d, d])

    if net_type == "unweighted":
        # Stage 1
        A_sum = np.sum(A, axis=0)
        ns_mats[0] = np.divide(A, A_sum, out=ns_mats[0], where=(A_sum!=0))
        A_i = A.copy()
        seen = np.eye(d)
        for i in range(1, r):
            seen = seen + A_i
            A_i = np.clip(A_i @ A, 0, 1)
            A_i[seen > 0] = 0
            A_sum = np.sum(A_i, axis=0)
            ns_mats[i] = np.divide(A_i, A_sum, out=ns_mats[i], where=(A_sum!=0))
    else:
        # Weighted or distance network
        A_binary = (A > 0).astype(float)
        if net_type == "distance":
            # Convert distances to connection weights: closer = stronger
            W = np.zeros_like(A, dtype=float)
            mask = A > 0
            W[mask] = 1.0 / A[mask]
        elif net_type == "weighted":
            W = A.copy()
        else:
            raise ValueError("net_type must be one of 'unweighted', 'weighted', or 'distance'")

        # Stage 1: direct neighbours with weights from W
        W_sum = np.sum(W, axis=0)
        ns_mats[0] = np.divide(W, W_sum, out=ns_mats[0], where=(W_sum != 0))

        # Binary tracking for stage determination, weighted tracking for weight accumulation
        B_i = A_binary.copy()
        W_i = W.copy()
        seen = np.eye(d)

        for i in range(1, r):
            seen = seen + B_i
            B_i = np.clip(B_i @ A_binary, 0, 1)
            W_i = W_i @ W
            B_i[seen > 0] = 0
            W_i[seen > 0] = 0
            W_sum = np.sum(W_i, axis=0)
            ns_mats[i] = np.divide(W_i, W_sum, out=ns_mats[i], where=(W_sum != 0))

    return ns_mats

def compute_neighbour_sums(ts: np.ndarray, ns_mats: np.ndarray, r: int) -> np.ndarray:
    """
    Compute the neighbour sums for each stage of neighbour dependence.

    Params:
        ts: np.array. Time series. Shape (n, d)
        ns_mats: np.array. Tensor of powers of the adjacency matrix. Shape (r, n, n)
        r: int. Maximum stage of neighbour dependence

    Returns:
        np.array. Time series and neighbour sums. Shape (n, d, 1 + r)
    """
    n, d = ts.shape
    data = np.zeros([n, d, 1 + r])
    data[:, :, 0] = ts
    data[:, :, 1:] = np.transpose(ts @ ns_mats, (1, 2, 0))
    return data

def weight_mats(ns_mats: np.ndarray, p: int, s: np.ndarray, d: int) -> np.ndarray:
    """
    Construct the matrices for mapping the gnar coefficients to var form. Also useful for the Yule-Walker equations.

    Params:
        ns_mats: np.array. Tensor of powers of the adjacency matrix. Shape (r, n, n)
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        np.array. Mapping matrices. Shape (d, p * d, p + sum(s))
    """
    W = np.zeros([d, p * d, p + np.sum(s)])
    w = np.vstack([np.eye(d).reshape(1, d, d), ns_mats]).transpose(2, 1, 0)
    s_tau = 0
    for i in range(p):
        W[:, i * d : (i + 1) * d, s_tau : s_tau + s[i] + 1] = w[:, :, :1 + s[i]]
        s_tau += s[i] + 1
    return W
