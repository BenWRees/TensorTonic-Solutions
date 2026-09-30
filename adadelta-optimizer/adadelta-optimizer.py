import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    """
    Returns a dictionary with new_w, new_E_grad_sq, and new_E_update_sq.
    """
    # Write code here
    w_array = np.array(w)
    grad_array = np.array(grad)
    E_grad_sq_array = np.array(E_grad_sq)
    E_update_sq_array = np.array(E_update_sq)
    
    running_sq_grad = rho * E_grad_sq_array + (1-rho) * grad_array ** 2

    param_change = -grad_array * np.sqrt((E_update_sq_array + eps)/(running_sq_grad + eps))

    E_update_change = rho * E_update_sq_array + (1-rho) * param_change**2 

    param_update = w_array + param_change 

    return {
        'new_w': param_update, 
        'new_E_grad_sq': running_sq_grad,
        'new_E_update_sq': E_update_change
    }