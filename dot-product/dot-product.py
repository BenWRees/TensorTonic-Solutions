import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    return float(np.sum([a*b for a,b in zip(x,y)]))