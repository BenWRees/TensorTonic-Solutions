def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    return [sum(values[i:i+window_size])/window_size for i in range(len(values)-window_size+1)]