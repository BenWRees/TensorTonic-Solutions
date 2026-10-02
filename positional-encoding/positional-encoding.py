import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pos = np.arange(seq_len)[:, None] 
    i = np.arange((d_model+1)//2)[None, :]

    angle = pos / base ** (2 * i/d_model)

    encoding = np.empty((seq_len, d_model))

    encoding[:, 0::2] = np.sin(angle)
    encoding[:, 1::2] = np.cos(angle[:, :encoding[:, 1::2].shape[1]])
    return encoding

    