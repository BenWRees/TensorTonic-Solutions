import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    # Write code here
    return float(np.sum([(fpr[i+1]-fpr[i])*(tpr[i]+tpr[i+1])/2 for i in range(len(fpr)-1)]))