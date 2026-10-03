def interaction_features(X: list) -> list:
    """
    Returns original features followed by unique pairwise products.
    """
    # Write code here
    result = []

    for x in X :
        expanded = x.copy() 

        for i in range(len(x)) :
            for j in range(i+1, len(x)) : 
                expanded.append(x[i] * x[j])

        result.append(expanded)
    return result

    