def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    # Write code here
    counts = dict() 
    for value in values :
        if value not in counts :
            counts[value] = 1 
            continue 
        counts[value] += 1

    total_values = sum([val for key, val in counts.items()])

    frequencies = dict([(k, v/total_values) for k,v in counts.items()])

    return [frequencies[a] for a in values]
