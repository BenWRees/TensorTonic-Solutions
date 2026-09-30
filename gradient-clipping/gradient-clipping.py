import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    # Write code here
    g_grad = np.array(g)
    grad_clipped = np.linalg.norm(g_grad.ravel(), ord=2) 
    return g_grad * (max_norm/grad_clipped) if grad_clipped > max_norm else g_grad