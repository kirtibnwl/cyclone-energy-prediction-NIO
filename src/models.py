"""ANN model factory matching the architecture from the paper.

Initial model:  MLPRegressor, hidden=(128, 64, 32), ReLU, Adam, lr=1e-3, alpha=1e-3
Optimised model: same architecture, trained only on features selected by
permutation importance.
"""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def build_ann(
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
    activation: str = "relu",
    solver: str = "adam",
    learning_rate_init: float = 1e-3,
    alpha: float = 1e-3,
    max_iter: int = 3000,
    random_state: int = 42,
) -> MLPRegressor:
    """Build the MLPRegressor used in the paper."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )
