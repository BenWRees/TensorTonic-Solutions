def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    # Write code here
    mean_series = sum(series)/len(series)
    var_series = sum([(a-mean_series)**2 for a in series])
    if var_series == 0 :
        return [1.0] + [0.0] * max_lag

    return [
         1.0 if i == 0 else (
            sum(
                (series[k] - mean_series) * (series[k + i] - mean_series)
                for k in range(len(series) - i)
            ) / var_series
            if var_series != 0 else 0.0
        )
        for i in range(max_lag + 1)
    ]