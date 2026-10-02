import torch

def muon_spectral_update(
    parameter: torch.Tensor, 
    gradient: torch.Tensor,
    previous_momentum: torch.Tensor, 
    momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum, orthogonalized_update.
    """
    new_momentum = momentum_coefficient * previous_momentum + gradient
    new_momentum_float = new_momentum.to(torch.float32)
    U, S, Vt = torch.linalg.svd(new_momentum_float, full_matrices=False)

    orthogonalized_update = U @ Vt
    orthogonalized_update = orthogonalized_update.to(parameter.dtype)
    new_parameter = parameter - learning_rate * orthogonalized_update 

    return {
        "new_parameter": new_parameter, 
        "new_momentum": new_momentum,
        "orthogonalized_update": orthogonalized_update
    }
