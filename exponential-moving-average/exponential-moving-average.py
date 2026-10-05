def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    EMA = [values[0]]
    for t in range(1,len(values)) : 
        EMA.append(alpha * values[t] + (1-alpha)*EMA[t-1])
    return EMA