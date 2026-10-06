def linear_interpolation(values: list) -> list:

    """

    Returns a copy with every missing value interpolated.

    """
    result = values.copy()
    for i, value in enumerate(values):
        if value is not None:
            continue

        left = find_left(i, values)
        right = find_right(i, values)
        
        result[i] = (
            values[left]
            + (i - left) / (right - left)
            * (values[right] - values[left])
        )

    return result

def find_left(i: int, x: list) -> float : 
    left = i - 1
    while left >= 0 and x[left] is None:
        left -= 1
    return left 

def find_right(i: int, x: list) -> float : 
    right = i + 1
    while right < len(x) and x[right] is None:
        right += 1
    return right 