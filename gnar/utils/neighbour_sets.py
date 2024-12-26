import numpy as np

def neighbour_set_mats(A, r):
    """
    Compute a tensor containing the powers of the adjacency matrix A up to stage s.

    Params:
        A: np.array. Adjacency matrix. Shape (n, n)
        r: int. Maximum stage of neighbour dependence
    
    Returns:
        ns_mats: np.array. Tensor of powers of the adjacency matrix. Shape (r, n, n)
    """
    d = A.shape[0]
    # Create the tensor containing the adjacency matrix for each stage of neighbour dependence up to stage r
    ns_mats = np.zeros([r, d, d])
    # Compute the stage 1 adjacency matrix
    A_sum = np.sum(A, axis=0)
    ns_mats[0] = np.divide(A, A_sum, out=ns_mats[0], where=(A_sum!=0))
    A_i = A.copy()
    # Initialise see matrix to keep track of nodes that have been visited to avoid cycles
    seen = np.eye(d)
    # Compute the adjacency matrix for each stage of neighbour dependence up to stage r
    for i in range(1, r):
        seen = seen + A_i
        A_i = np.clip(A_i @ A, 0, 1)
        A_i[seen > 0] = 0
        A_sum = np.sum(A_i, axis=0)
        ns_mats[i] = np.divide(A_i, A_sum, out=ns_mats[i], where=(A_sum!=0))
    return ns_mats

def compute_neighbour_sums(ts, ns_mats, r):
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