import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    new_s = running_avg_update(beta, s, g)
    new_w = parameter_update(w, new_s, g, lr, eps)

    new_s_list = new_s.tolist()
    new_w_list = new_w.tolist()
    
    return (new_w_list, new_s_list)

def running_avg_update(beta: float, RMS: np.ndarray, grad: np.ndarray) -> np.ndarray :
    return beta * RMS + (1-beta) * grad ** 2

def parameter_update(param: np.ndarray, RMS: np.ndarray, grad: np.ndarray, lr: float, eps: float) -> np.ndarray :
    lr_enhanced = lr / (np.sqrt(RMS + eps))
    return param - lr_enhanced * grad 