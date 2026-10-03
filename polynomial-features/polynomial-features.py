def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """
    # Write code here
    result = [] 
    for val in values : 
        powers = [val**d for d in range(0,degree+1)] 
        result.append(powers)
    return result