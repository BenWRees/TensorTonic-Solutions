import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Write code here
    w_array = np.array(w)
    m_array = np.array(m)
    v_array = np.array(v)
    grad_array = np.array(grad)
    
    first_moment = beta1 * m_array + (1-beta1) * grad_array 
    second_moment = beta2 * v_array + (1-beta2)* grad_array**2 

    adjusted_first_moment = beta1 * first_moment + (1-beta1) * grad_array

    new_weight = w_array - lr * adjusted_first_moment/(np.sqrt(second_moment)+eps)

    return {
        "new_w": new_weight,
        "new_m": first_moment,
        "new_v": second_moment
    }