# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from utils import seed_everything
from metrics import SmoothBCELogits, LabelSmoothingLoss
from models import BaselineModel

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


class FineTuneScheduler:
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.epochs_per_step: int = 0
        self.frozen_layers: list = []

    def copy_without_top(
        self,
        model: nn.Module,
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
    """Main training loop 
    Args:
        input_path (Path): data/processed
        output_path (Path): models
    """
    pass


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    seed_everything(SEED)

    # TODO: implement training logic

    main()
