import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    # Write code here
    return float(np.sqrt(np.sum([(a-b)**2 for a,b in zip(x,y)])))