import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def _binary_cross_entropy_loss(p: np.ndarray, y: np.ndarray) -> float :
    n = p.shape 
    entropy_loss = np.sum(y*np.log(p)+(1-y)*np.log(p))
    return - entropy_loss/n

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m,n = X.shape
    weights = np.zeros((n,))
    bias = 0.0

    for _ in range(steps) : 
        z = X @ weights + bias 
        p = _sigmoid(z)
        loss = _binary_cross_entropy_loss(p, y)
        dL_w = (X.T @ (p-y))/m
        dL_b = np.sum(p-y)/m
        weights = weights - lr * dL_w 
        bias = bias - lr * dL_b 

    return weights, bias
        
        




