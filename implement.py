import numpy as np
from random import seed as set_seed
from sklearn.metrics import pairwise_distances


def construct_neighbourhood_graph(X, metric='euclidean', type='k', hp=3):

    """
    Constructs a neighborhood graph from a dataset based on either k-NN or ε-NN.

    Inputs:
    -----------
    X : ndarray of shape (n_samples, n_dim)
        The input data.

    metric : str, optional, default='euclidean'
        The distance metric to use for computing pairwise distances.

    type : str, optional, default='k'
        The type of neighborhood graph to construct:
        - 'k' : Uses k-NN with hp as the number of neighbours.
        - 'eps' : Uses ε-NN with hp as the distance threshold.

    hp : int or float, optional, default=3
        The hyperparameter controlling neighborhood selection:
        - If type='k', hp is the number of nearest neighburs for k-NN.
        - If type='eps', hp is the distance threshold for ε-NN.

    Outputs:
    --------
    G : ndarray of shape (n_samples, n_samples)
        The symmetric neighborhood graph represented as a weighted adjacency matrix, where:
        - G[i, j] is the distance/edge weight between vertices i and j if they are neighbours.
        - Inf indicates no edge.
    """

    n = X.shape[0]

    pwD = pairwise_distances(X, metric=metric)

    G = np.full((n, n), np.inf)

    # Determine indices of neighbours:
    if type=='k':
        knn_indices = np.argsort(pwD, axis=1)[:, 1:hp+1]

        row_indices = np.repeat(np.arange(n), hp)
        col_indices = knn_indices.ravel()
    elif type=='eps':
        mask = (pwD < hp) & (pwD > 0)
        row_indices, col_indices = np.where(mask)

    distances = pwD[row_indices, col_indices]

    # Set weights of neighbours as distance:
    G[row_indices, col_indices] = distances
    G[col_indices, row_indices] = distances

    return G


def construct_shortest_distance_graph(G):

    """
    Computes the shortest path distances between all pairs of vertices in the graph
    using the Floyd-Warshall algorithm.

    Inputs:
    -----------
    G : ndarray of shape (n_samples, n_samples)
        The symmetric neighborhood graph represented as a weighted adjacency matrix, where:
        - G[i, j] is the distance/edge weight between vertices i and j if they are neighbours.
        - Inf indicates no edge.

    Outputs:
    --------
    D : ndarray of shape (n_samples, n_samples)
        The shortest path distance matrix, where D[i, j] gives the shortest 
        distance from vertex i to vertex j. D will be symmetric.
    """

    n = G.shape[0]
    D = G.copy()

    # Iterate over all vertices to update shortest paths:
    for v in range(n):
        D = np.minimum(D, D[:, v][:, np.newaxis] + D[v, :])

    return D


def construct_low_dim_embedding(D, d=2, seed=2201):

    """
    Constructs a low-dimensional embedding using  Classical Multidimensional Scaling (MDS).

    Inputs
    ----------
    D : ndarray of shape (n_samples, n_samples)
        A symmetric distance matrix.

    d : int, optional (default=2)
        The dimensionality of the low-dimension embedding.

    seed : int, optional (default=2201)
        Random seed for reproducibility.

    Outputs
    -------
    Y : ndarray of shape (n_samples, d)
        The resultant d-dimensional embedding after performing Classical MDS.
    """
     
    n = D.shape[0]
    D_squared = D ** 2

    # Compute Gram matrix after double-centering:
    C = np.eye(n) - np.ones((n, n)) / n
    tau = -0.5 * (C @ D_squared @ C)

    # Perform eigendecomposition:
    set_seed(seed)
    eigvals, eigvecs = np.linalg.eig(tau)

    idx = np.argsort(eigvals)[::-1] 
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top d eigenvectors to construct embedding:
    L = np.diag(np.sqrt(eigvals[:d]))
    Y = eigvecs[:, :d] @ L 

    return Y


def isomap(X, type='k', hp=3, d=2, seed=2201):
    """
    Implements the ISOMAP dimension reduction algorithm.

    Steps:
    1. Constructs a neighborhood graph based on Euclidean distances using either k-NN ε-NN.
    2. Constructs the shortest path distance matrix using the Floyd-Warshall algorithm.
    3. Constructs a low-dimensional embedding using Classical Multidimensional Scaling (MDS).

    Inputs
    ----------
    X : ndarray of shape (n_samples, n_dim)
        The input data.

    d : int, optional (default=2)
        The dimensionality of the low-dimension embedding.

    type : str, optional, default='k'
        The type of neighborhood graph to construct:
        - 'k' : Uses k-NN with hp as the number of neighbours.
        - 'eps' : Uses ε-NN with hp as the distance threshold.

    hp : int or float, optional, default=3
        The hyperparameter controlling neighborhood selection:
        - If type='k', hp is the number of nearest neighburs for k-NN.
        - If type='eps', hp is the distance threshold for ε-NN.

    seed : int, optional (default=2201)
        Random seed for reproducibility.

    Outputs
    -------
    G : ndarray of shape (n_samples, n_samples)
        The neighborhood graph represented as a weighted adjacency matrix.
    
    D : ndarray of shape (n_samples, n_samples)
        The shortest path distance matrix.
    
    Y : ndarray of shape (n_samples, d)
        The resultant d-dimensional embedding after performing Classical MDS.
    """

    G = construct_neighbourhood_graph(X, metric='euclidean', type=type, hp=hp)
    D = construct_shortest_distance_graph(G)
    Y = construct_low_dim_embedding(D, d=d, seed=seed)
    
    return G, D, Y

