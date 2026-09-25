import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X_arr = np.array(X)
    cov_matrix = compute_cov_matrix(X_arr)
    std_dev = compute_std_dev(cov_matrix)
    
    
    
    return cov_matrix / np.outer(std_dev, std_dev)

def centre_data(X: np.ndarray) -> np.ndarray :
    """
        Centre the column means 
    """
    X_centred = np.zeros(X.shape)
    for i in range(X.shape[1]) :
        X_centred[:,i] = X[:, i] - np.mean(X[:, i])

    return X_centred

def compute_cov_matrix(X: np.ndarray) -> np.ndarray : 
    """
        Compute the covariance matrix from X 
        Params:
            X (np.ndarray): centred dataset with shape (N,D). 
                            N -> number of features 
                            D -> dimension of each feature 
        Returns: 
            np.ndarray: The DxD covariance matrix 
    """
    N, D = X.shape 
    X_centre = centre_data(X)
    return (X_centre.T @ X_centre)/(N-1)
    

def compute_std_dev(cov: np.ndarray) -> np.ndarray : 
    """
        Compute the standard deviation of a dataset 
        Params:
            cov (np.ndarray): The (D,D) covariance matrix 

        Returns: 
            np.ndarray: the standard deviation of the (N,D) dataset
    """
    std_dev = np.sqrt(np.diag(cov)).flatten()
    for t,x in enumerate(std_dev) :
        if x == 0 : 
            std_dev[t] = np.nan
    return np.array(std_dev)
            

