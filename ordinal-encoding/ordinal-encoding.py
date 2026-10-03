def ordinal_encoding(values: list, ordering: list) -> list:
    """
    Returns the ordinal index of every input value.
    """
    # Write code here
    positions = {value: i for i,value in enumerate(ordering)}
    return [positions[value] for value in values]