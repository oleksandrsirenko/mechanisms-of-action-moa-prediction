# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from utils import seed_everything

import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn
import torch.nn.functional as F


# CONSTANTS
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 16
BATCH_SIZE = 128

WEIGHT_DECAY = {"ALL_TARGETS": 1e-5, "SCORED_ONLY": 3e-6}
MAX_LR = {"ALL_TARGETS": 1e-2, "SCORED_ONLY": 3e-3}
DIV_FACTOR = {"ALL_TARGETS": 1e3, "SCORED_ONLY": 1e2}
PCT_START = 0.1


class MoaDataset(torch.Dataset):
    def __init__(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        super().__init__()
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[torch.float]:
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }


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


class BaselineModel(nn.Module):
    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_sizes: List[int] = [2048, 1024, 1024, 512]
        self.dropouts: List[float] = [0.3, 0.3, 0.3, 0.3]

        self.batch_norm_1 = nn.BatchNorm1d(self.n_features)
        self.dense_1 = nn.Linear(self.n_features, self.hidden_sizes[0])

        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dropout_2 = nn.Dropout(self.dropouts[0])
        self.dense_2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.batch_norm_3 = nn.BatchNorm1d(self.hidden_sizes[1])
        self.dropout_3 = nn.Dropout(self.dropouts[1])
        self.dense_3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])

        self.batch_norm_4 = nn.BatchNorm1d(self.hidden_sizes[2])
        self.dropout_4 = nn.Dropout(self.dropouts[2])
        self.dense_4 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])

        self.batch_norm_5 = nn.BatchNorm1d(self.hidden_sizes[3])
        self.dropout_5 = nn.Dropout(self.dropouts[3])
        self.dense_5 = nn.Linear(self.hidden_sizes[3], self.n_targets)

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x


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


class FineTuneScheduler:
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.epochs_per_step: int = 0
        self.frozen_layers: list = []

    def copy_without_top(
        self,
        model: BaselineModel,
        num_features: int,
        num_targets: int,
        num_targets_new: int,
    ) -> nn.Module:

        self.frozen_layers = []
        fine_model = BaselineModel(num_features, num_targets)
        fine_model.load_state_dict(model.state_dict())

        # Freeze all weights
        for name, param in fine_model.named_parameters():
            layer_index = name.split(".")[0][-1]

            if layer_index == 5:
                continue

            param.requires_grad = False

            # Save frozen layer names
            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = self.epochs // len(self.frozen_layers)

        # Replace the top layers with another ones
        fine_model.batch_norm5 = nn.BatchNorm1d(fine_model.hidden_size[3])
        fine_model.dropout5 = nn.Dropout(fine_model.dropout_value[3])
        fine_model.dense5 = nn.utils.weight_norm(
            nn.Linear(fine_model.hidden_size[-1], num_targets_new)
        )
        fine_model.to(DEVICE)
        return fine_model

    def step(self, epoch: int, model: BaselineModel) -> None:
        if len(self.frozen_layers) == 0:
            return

        if epoch % self.epochs_per_step == 0:
            last_frozen_index = self.frozen_layers[-1]

            # Unfreeze parameters of the last frozen layer
            for name, param in model.named_parameters():
                layer_index = name.split(".")[0][-1]

                if layer_index == last_frozen_index:
                    param.requires_grad = True

            # Remove the last layer as unfrozen
            del self.frozen_layers[-1]


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path: Path, output_path: Path) -> None:
    """Main training loop containing following steps:
    1. Read processed dataset from `data/processed`
    2. Make training and validation splits / kfolds 
    3. Create Dataset fot PyTorch model
    4. Create Dataloaders
    5. Model Training 
    6. Save models to `models` directory

    Args:
        input_path (Path): data/processed
        output_path (Path): models
    """
    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    seed_everything(SEED)

    # TODO: implement training loop logic

    main()
