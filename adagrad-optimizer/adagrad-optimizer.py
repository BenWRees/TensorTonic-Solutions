import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    w_array = np.array(w)
    g_array = np.array(g)
    G_array = np.array(G)

    new_G = G_array + g_array**2 
    new_w = w_array - lr * g_array/(np.sqrt(new_G + eps))
    return {
        'new_w': new_w,
        'new_G': new_G
    }