import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    var: float = compute_variance(x)
    std_dev: float = np.sqrt(var, dtype=float) 
    return {'variance': float(var), 'standard_deviation': float(std_dev)}
    

def compute_mean(x: list) -> float : 
    return np.mean(np.array(x, dtype=float))

def compute_variance(x: list) -> float :
    x_length = len(x)
    mean_x = compute_mean(x)
    centred_x = np.sum(np.array([(x_i-mean_x)**2 for x_i in x], dtype=float))
    return 1/(x_length-1)*centred_x