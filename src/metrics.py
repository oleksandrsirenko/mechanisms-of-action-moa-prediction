"""Custom metrics for PyTorch. Need to fix this."""

import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class SmoothBCELogits(_WeightedLoss):
    def __init__(
        self, weight: float = None, reduction: str = "mean", smoothing: float = 0.0
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.smoothing = smoothing

    @staticmethod
    def _smooth(
        targets: torch.float, n_lables: int, smoothing: float = 0.0
    ) -> torch.float:
        assert 0 <= smoothing < 1

        with torch.no_grad():
            targets = targets * (1 - smoothing) + 0.5 * smoothing

        return targets

    def forward(self, inputs: torch.float, targets: torch.float) -> torch.float:
        targets = SmoothBCELogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_labels: int, smoothing: float = 0.0, dim: int = -1) -> None:
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_labels
        self.dim = dim

    def forward(self, pred: torch.float, target: torch.float) -> torch.float:
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
