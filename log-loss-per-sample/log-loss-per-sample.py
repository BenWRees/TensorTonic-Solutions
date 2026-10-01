import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here

    #clip each probability 
    y_pred_clipped = [min(1-eps, max(eps, y)) for y in y_pred]

    return [-(y*math.log(p)+(1-y)*math.log(1-p)) for y,p in zip(y_true, y_pred_clipped)]

    