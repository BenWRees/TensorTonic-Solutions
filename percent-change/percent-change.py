def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    # Write code here
    return [(series[i+1]-series[i])/series[i] if series[i] !=0 else 0.0 for i in range(len(series)-1)]