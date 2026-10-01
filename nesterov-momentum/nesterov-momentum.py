import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    # Write code here
    w_array = np.array(w)
    v_array = np.array(v)
    grad_array = np.array(grad)

    velocity = momentum * v_array + lr * grad_array 
    parameter = w_array - velocity
    return {
        'new_w': parameter, 
        'new_v': velocity
    }