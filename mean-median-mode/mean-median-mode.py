from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x_length = len(x)
    sorted_x = sorted(x)

    mean: float = compute_mean(x, x_length)
    mode: float = compute_mode(x)
    median:float = compute_median(sorted_x, x_length)

    return {'mean': mean, 'median': median, 'mode': mode}

def compute_mean(x: list, length: int) -> float :
    return float(np.sum([x_i for x_i in x])/ length)

def compute_mode(x: list) -> float : 
    freq = [(a, x.count(a)) for a in x]
    max_freq = max(freq, key=lambda x: x[1])[1]
    largest_freqs = [t for t in freq if t[1] == max_freq]
    if len(largest_freqs) > 1 : 
        mode = min(largest_freqs, key=lambda x : x[0])[0]
    else : 
        mode = largest_freqs[0][0]
    return float(mode)

def compute_median(x: list, length: int) -> float :
    if length != 0 :
        median = float((x[length//2-1]/2.0+x[length//2]/2.0, x[length//2])[length % 2])
    else : 
        raise ValueError("List must have elements!") 
    return median
        
