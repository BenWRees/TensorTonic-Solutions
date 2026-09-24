def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    grad = lambda x : 2*a*x+b
    step_fn = lambda sol, t : sol[t-1] - lr* grad(sol[t-1])
    solutions = {0: x0}
    for t in range(1,steps+1) :
        solutions[t] = step_fn(solutions, t)
    return solutions[steps]
        