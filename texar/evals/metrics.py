"""
Various metrics.
"""
import torch

__all__ = [
    "accuracy",
]


def accuracy(labels: torch.Tensor, preds:torch.Tensor):
    """Calculates the accuracy of predictions.

    Args:
        labels: The ground truth values. A Tensor of the same shape of
            :attr:`preds`.
        preds: A Tensor of any shape containing the predicted values.

    Returns:
        A float scalar Tensor containing the accuracy.
    """
    return (preds == labels).float().mean()