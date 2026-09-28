import math

def cosine_annealing_schedule(base_lr: float, min_lr: float, total_steps: int, current_step: int) -> float:
    """
    Returns the cosine-annealed learning rate for the requested step.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if not 0 <= current_step <= total_steps:
        raise ValueError("current_step must be between 0 and total_steps")

    return min_lr + 0.5 * (base_lr - min_lr) * (
        1 + math.cos(math.pi * current_step / total_steps)
    )