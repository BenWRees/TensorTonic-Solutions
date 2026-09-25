import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X_arr = np.array(X) 
    N,D = X_arr.shape
    X_arr_centred = X_arr - np.mean(X, axis=0)
    return (X_arr_centred.T @ X_arr_centred)/(N-1)