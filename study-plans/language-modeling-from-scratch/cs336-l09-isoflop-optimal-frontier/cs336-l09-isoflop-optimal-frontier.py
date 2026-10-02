import torch

def isoflop_optimal_frontier(compute_budgets: torch.Tensor, parameter_counts: torch.Tensor, terminal_losses: torch.Tensor) -> dict:
    """
    Returns a dict: optima (list of dicts), frontier_scale, frontier_exponent.
    """
    optima = [] 

    for i in range(compute_budgets.shape[0]) :
        N = parameter_counts[i]
        L = terminal_losses[i]

        x = torch.log(N)

        X = torch.stack((x.square(), x, torch.ones_like(x)), dim=1)

        coeffs = torch.linalg.lstsq(X,L).solution 

        a,b,c = coeffs 

        x_star = - b/(2.0*a)
        N_star = torch.exp(x_star)

        C = compute_budgets[i]
        D_star = C / (6.0 * N_star)

        L_star = a * x_star.square() + b * x_star + c 

        optima.append({
            'compute': float(C.item()),
            'parameters': float(N_star.item()),
            'tokens': float(D_star.item()),
            'loss': float(L_star.item())
        })

    compute = torch.stack([
        compute_budgets[i] for i in range(compute_budgets.shape[0])
    ])

    parameters = torch.stack([
        torch.as_tensor(optima[i]["parameters"], dtype=torch.float64, device=compute_budgets.device)
        for i in range(len(optima))
    ])

    log_C = torch.log(compute)
    log_N = torch.log(parameters)

    frontier_X = torch.stack(
        (torch.ones_like(log_C), log_C), 
        dim=1
    )

    frontier_coeffs = torch.linalg.lstsq(
        frontier_X, 
        log_N, 
    ).solution 

    log_k, alpha = frontier_coeffs
    k = torch.exp(log_k)

    return {
        "optima": optima, 
        "frontier_scale": float(k.item()),
        "frontier_exponent": float(alpha.item())
    }
