import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    n = A.shape[0]

    # Form [A | I]
    A_aug = np.eye(n)

    # Gauss-Jordan elimination
    for col in range(n):
        pivot_row = col + np.argmax(np.abs(A[col:, col]))

        # If the pivot is zero, A is singular
        if np.isclose(A[pivot_row, col], 0):
            return None

        # Swap pivot row into position
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            A_aug[[col, pivot_row]] = A_aug[[pivot_row, col]]

        pivot = A[col, col]
        A[col] /= pivot
        A_aug[col] /= pivot

        factors = A[:, col].copy()
        factors[col] = 0

        A -= factors[:, None] * A[col]
        A_aug -= factors[:, None] * A_aug[col]

    return A_aug
    