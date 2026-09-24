import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    return np.sum(np.array([x_i * p_i for x_i,p_i in zip(x,p)], dtype=float))