import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if seqs == [] : 
        return np.empty((0,0), dtype=int)
    
    if max_len is not None : 
        padded_seq = create_padded_seq(seqs, pad_value, max_len)

    else : 
        max_len = len(max(seqs, key=len))
        padded_seq = create_padded_seq(seqs, pad_value, max_len)
    
    return np.array(padded_seq)


def create_padded_seq(seqs: list, pad_value: int, max_len: int) -> list :
    padded_seq = []
    for x in seqs : 
        if len(x) < max_len :
            vals_to_pad = max_len-len(x)
            padding = [pad_value] * vals_to_pad
            x.extend(padding)
            padded_seq.append(x)
        
        elif len(x) > max_len :
            padded_seq.append(x[:max_len])
        else : 
            padded_seq.append(x)
    return padded_seq
                
                