import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    grad = np.array(grad)
    param = np.array(param)
    
    moment1: np.ndarray = moment_calculation(beta1, grad, np.array(m), 1)
    moment2: np.ndarray = moment_calculation(beta2, grad, np.array(v), 2)
    moment1_corrected: np.ndarray = moment_correction(moment1, beta1, t)
    moment2_corrected: np.ndarray = moment_correction(moment2, beta2, t)
    
    descent: np.ndarray = moment1_corrected / (np.sqrt(moment2_corrected)+eps)
    
    param_update: np.ndarray = param - lr * descent 
    
    return (param_update, moment1, moment2)

def moment_calculation(beta: float, grad: np.ndarray, moment: np.ndarray, order: int) -> np.ndarray :
    if order == 1 :
        moment = beta * moment + (1-beta) * grad
    elif order == 2 :
        moment = beta * moment + (1-beta) * grad**2
    else :
        raise ValueError("Wrong Moment")
    return moment

def moment_correction(moment: np.ndarray, beta: float, t: int) -> np.ndarray :
    return moment / (1-beta**t)