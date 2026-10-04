def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    return [[series[t - lag] for lag in lags] for t in range(max(lags), len(series))]