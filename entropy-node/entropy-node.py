import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    classes = {}
    for node in y : 
        if node in classes : 
            classes[node] = classes[node]+1
        else :
            classes[node] = 1

    n_classes = len(classes.keys())

    probabilities = {}
    for cls, count in classes.items() : 
        probabilities[cls] = count/len(y)


    entropy_per_item = []
    for cls, probs in probabilities.items() : 
        if probs == 0.0 :
            entropy_per_item.append(0.0)
            continue 
        entropy_per_item.append(probs*np.log2(probs)) 


    return float(-np.sum(entropy_per_item))
        
        