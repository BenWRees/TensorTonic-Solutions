import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X_arr = np.array(X)
    X_arr_centre = X_arr - np.mean(X_arr, axis=0)
    cov_matrix = covariance_matrix(X_arr_centre)
    eigenvectors = obtain_top_k_eignvectors(cov_matrix, k)
    return X_arr_centre @ eigenvectors


def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    N,D = X.shape
    return (X.T @ X)/(N-1)

def obtain_top_k_eignvectors(X: np.ndarray, k: int) -> np.ndarray :
    """
        Obtain the top k eigenvectors of matrix X sorted by eigenvalue in descending order. 
        Params:
            X (np.ndarray) : Matrix of shape (D,D)
            k (int): number of eigenvectors 
        Returns:
            np.ndarray: Matrix of shape (D,k) of the top k eigenvectors 
    """
    eigenvalues, eigenvectors = np.linalg.eig(X)
    combined_eigs = [
        (eigenvalues[i], eigenvectors[:, i])
        for i in range(len(eigenvalues))
    ]
    sorted_eigs = sorted(combined_eigs, key=lambda x: x[0], reverse=True)[:k]
    return np.column_stack([a[1] for a in sorted_eigs])