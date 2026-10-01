def catalog_coverage(recommendations: list, n_items: int) -> float:
    """
    Returns the fraction of catalog items that were recommended.
    """
    # Write code here
    return len(_unique_list(recommendations))/n_items if n_items > 0 else 0.0


def _unique_list(x: list) -> set : 
    set_of_x = set() 

    x_flatten = [item for sublist in x for item in sublist]

    for itm in x_flatten : 
        set_of_x.add(itm)
    return set_of_x