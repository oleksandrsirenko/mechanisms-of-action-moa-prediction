"""
Script for training a model for the Mechanisms of Action (MoA) prediction task.
This script uses a simple feed-forward neural network model, PyTorch as the backend, and cross-validation for better model performance.
"""
import json
import click
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter

# setting up logs
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter("logs")

# Define constants
INPUT_PATH = "data/processed"
OUTPUT_PATH = "models"
BATCH_SIZE = 64
LR = 0.001
N_EPOCHS = 2
FOLDS = [0, 1, 2, 3, 4]  # Folds to use for cross-validation
MODEL_CONFIG = {"layer1_size": 1024, "layer2_size": 2048}


def prepare_data(input_path: Path, fold_id: int):
    train_df: pd.DataFrame = pd.read_csv(input_path / f"train_fold{fold_id}.csv")
    valid_df: pd.DataFrame = pd.read_csv(input_path / f"valid_fold{fold_id}.csv")

    # Drop column 1 which only has "trt_cp" values
    train_df = train_df.drop(columns=train_df.columns[1])
    valid_df = valid_df.drop(columns=valid_df.columns[1])

    # Separate features and targets
    train_features: np.ndarray = train_df.iloc[:, 1:875].values
    train_targets: np.ndarray = train_df.iloc[:, 875:].values

    valid_features: np.ndarray = valid_df.iloc[:, 1:875].values
    valid_targets: np.ndarray = valid_df.iloc[:, 875:].values

    return train_features, train_targets, valid_features, valid_targets


class MoaDataset(Dataset):
    def __init__(self, features: np.array, targets: np.array):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dct = {
            "x": torch.tensor(self.features[idx, :], dtype=torch.float),
            "y": torch.tensor(self.targets[idx, :], dtype=torch.float),
        }
        return dct


class MoaModel(nn.Module):
    def __init__(
        self, num_features: int, num_targets: int, layer1_size: int, layer2_size: int
    ):
        super(MoaModel, self).__init__()
        self.layer1 = nn.Linear(num_features, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, num_targets)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(inputs))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# ResNet model
class ResBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout_rate: float = 0.0
    ) -> None:
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        self.lin2 = nn.Linear(out_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.lin1(x))
        out = self.dropout(out)
        out = self.lin2(out)
        out += residual
        return out


class ResNet(nn.Module):
    def __init__(self, num_features: int, num_targets: int, layer_sizes: List[int]):
        super(ResNet, self).__init__()
        layers = []
        layers.append(nn.Linear(num_features, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            layers.append(ResBlock(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.Linear(layer_sizes[-1], num_targets))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs = data["x"].to(device)
        targets = data["y"].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    total_loss = 0

    for data in dataloader:
        inputs = data["x"].to(device)
        targets = data["y"].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss


def run_training(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    model_path: str,
):
    best_valid_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_dataloader, criterion, optimizer, device)
        valid_loss = evaluate_model(model, valid_dataloader, criterion, device)
        logging.info(
            f"Epoch {epoch+1} / {num_epochs} - Train Loss: {train_loss} - Valid Loss: {valid_loss}"
        )

        # Write to tensorboard
        writer.add_scalars("Loss", {"train": train_loss, "valid": valid_loss}, epoch)

        # learning rate scheduler
        scheduler.step(valid_loss)

        # Save model parameters if validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": valid_loss,
                },
                model_path,
            )
            logging.info(
                f"Saved model parameters to {model_path}. Validation loss: {valid_loss}"
            )


def make_model(
    model_name: str, num_features: int, num_targets: int, model_config: Dict
):
    if model_name == "MoaModel":
        model = MoaModel(
            num_features,
            num_targets,
            model_config["layer1_size"],
            model_config["layer2_size"],
        )
    elif model_name == "ResNet":
        model = ResNet(num_features, num_targets, model_config["layer_sizes"])
    else:
        raise Exception(f"Unknown model: {model_name}")
    return model


@click.command()
@click.option(
    "--model_name", type=click.Choice(["MoaModel", "ResNet"], case_sensitive=False)
)
@click.option("--config", type=click.Path(exists=True))
def main(model_name: str, config: str):
    for fold_id in FOLDS:
        train_features, train_targets, valid_features, valid_targets = prepare_data(
            Path(INPUT_PATH), fold_id
        )

        # Now use this data with the Dataset, Dataloader and Model
        train_dataset = MoaDataset(train_features, train_targets)
        valid_dataset = MoaDataset(valid_features, valid_targets)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_features = train_features.shape[1]
        num_targets = train_targets.shape[1]
        # print(f"Number of features: {num_features}")
        # print(f"Number of targets: {num_targets}")

        # Load model configuration
        with open(config, "r") as f:
            model_config = json.load(f)

        model = make_model(model_name, num_features, num_targets, model_config)
        model.to(device)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        run_training(
            model,
            train_dataloader,
            valid_dataloader,
            criterion,
            optimizer,
            device,
            num_epochs=N_EPOCHS,
            model_path=f"{OUTPUT_PATH}/{model_name}_fold{fold_id}.pth",
        )


if __name__ == "__main__":
    main()
