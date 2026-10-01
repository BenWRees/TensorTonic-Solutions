import torch


def _forward(
    inputs: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute predictions and store intermediate layer activations."""
    H = inputs
    activations = [H]

    for W, b in zip(weights, biases):
        H = torch.tanh(H @ W.T + b)
        activations.append(H)

    return H[:, 0], activations


def _loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the summed squared-error loss."""
    return ((predictions - targets) ** 2).sum()


def _backward(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    activations: list[torch.Tensor],
    weights: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute gradients of the loss with respect to all parameters."""
    dH = (2 * (predictions - targets)).unsqueeze(1)

    weight_gradients = [None] * len(weights)
    bias_gradients = [None] * len(weights)

    for layer in range(len(weights) - 1, -1, -1):
        W = weights[layer]
        H_prev = activations[layer]
        H = activations[layer + 1]

        # Derivative through tanh.
        dZ = dH * (1 - H * H)

        # Parameter gradients.
        weight_gradients[layer] = dZ.T @ H_prev
        bias_gradients[layer] = dZ.sum(dim=0)

        # Propagate gradient to the previous layer.
        dH = dZ @ W

    return weight_gradients, bias_gradients


def _update_parameters(
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    weight_gradients: list[torch.Tensor],
    bias_gradients: list[torch.Tensor],
    learning_rate: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Apply one gradient-descent update to all parameters."""
    weights = [
        W - learning_rate * dW
        for W, dW in zip(weights, weight_gradients)
    ]
    biases = [
        b - learning_rate * db
        for b, db in zip(biases, bias_gradients)
    ]

    return weights, biases


def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int,
) -> tuple:
    """
    Train a scalar-output tanh MLP using full-batch gradient descent.

    Returns the final predictions, final loss, trained weights, trained
    biases, and the loss recorded immediately before each update.
    """
    learning_rate = float(learning_rate)

    trained_weights = [W.clone() for W in weights]
    trained_biases = [b.clone() for b in biases]
    loss_history = []

    for _ in range(steps):
        predictions, activations = _forward(
            inputs,
            trained_weights,
            trained_biases,
        )

        loss = _loss(predictions, targets)
        loss_history.append(loss.clone())

        weight_gradients, bias_gradients = _backward(
            predictions,
            targets,
            activations,
            trained_weights,
        )

        trained_weights, trained_biases = _update_parameters(
            trained_weights,
            trained_biases,
            weight_gradients,
            bias_gradients,
            learning_rate,
        )

    predictions, _ = _forward(
        inputs,
        trained_weights,
        trained_biases,
    )
    loss = _loss(predictions, targets)

    return (
        predictions,
        loss,
        trained_weights,
        trained_biases,
        loss_history,
    )