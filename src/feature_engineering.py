"""
Basic feature engineering. Includes transformation, scaling, 
and feature selection. Works but needs review, improvements,
refactoring, and adaptation for transformer models.
"""
import click
import logging
from pathlib import Path
import pandas as pd
import time
import psutil
from utils import seed_everything

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    KBinsDiscretizer,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

SEED = 42
THRESHOLD = 0.9


def make_transform_pipe(n_pca: int, n_qt: int = 100, n_kbd: int = 10) -> FeatureUnion:
    """
    Create a pipeline of transformations with FeatureUnion.

    Args:
        n_pca (int): Number of Principal Components for PCA.
        n_qt (int, optional): Number of quantiles for QuantileTransformer. Defaults to 100.
        n_kbd (int, optional): Number of bins for KBinsDiscretizer. Defaults to 10.

    Returns:
        FeatureUnion: Composite estimator applying a list of transformer objects in parallel to the input data.
    """
    transforms = [
        ("mms", MinMaxScaler()),
        ("ss", StandardScaler()),
        ("rs", RobustScaler()),
        ("qt", QuantileTransformer(n_quantiles=n_qt, output_distribution="normal")),
        ("kbd", KBinsDiscretizer(n_bins=n_kbd, encode="ordinal", strategy="uniform")),
        ("pca", PCA(n_components=n_pca)),
    ]

    return FeatureUnion(transforms)


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load raw data from csv files.

    Args:
        input_path (Path): Path to the input data directory.

    Returns:
        pd.DataFrame: Raw data loaded from csv files.
    """
    logging.info("Reading raw data...")
    return (
        pd.read_csv(input_path / "train_features.csv"),
        pd.read_csv(input_path / "train_targets_scored.csv"),
        pd.read_csv(input_path / "test_features.csv"),
        pd.read_csv(input_path / "sample_submission.csv"),
    )


def process_data(input_path: Path, output_path: Path) -> None:
    """
    Load and process raw data, then save the processed data to csv files.

    Args:
        input_path (Path): Path to the input data directory.
        output_path (Path): Path to the output data directory.
    """
    start_time = time.time()
    process_start_time = time.process_time()

    train_features_df, train_targets_df, test_features_df, test_targets_df = load_data(
        input_path
    )
    # Targets without changing
    GENES = [col for col in train_features_df.columns if col.startswith("g-")]
    CELLS = [col for col in train_features_df.columns if col.startswith("c-")]

    # Transformaitons
    gene_trans_pipe = make_transform_pipe(n_pca=600)
    cell_trans_pipe = make_transform_pipe(n_pca=50)

    train_gene_features = gene_trans_pipe.fit_transform(train_features_df[GENES])
    train_cell_features = cell_trans_pipe.fit_transform(train_features_df[CELLS])

    test_gene_features = gene_trans_pipe.transform(test_features_df[GENES])
    test_cell_features = cell_trans_pipe.transform(test_features_df[CELLS])

    TRANS_GENES = [f"g-{i}" for i in range(train_gene_features.shape[1])]
    TRANS_CELLS = [f"c-{i}" for i in range(train_cell_features.shape[1])]

    train_gene_df = pd.DataFrame(train_gene_features, columns=TRANS_GENES)
    train_cell_df = pd.DataFrame(train_cell_features, columns=TRANS_CELLS)

    test_gene_df = pd.DataFrame(test_gene_features, columns=TRANS_GENES)
    test_cell_df = pd.DataFrame(test_cell_features, columns=TRANS_CELLS)

    # Concatinate gene and cell features
    train_df = pd.concat(
        [train_gene_df.reset_index(drop=True), train_cell_df.reset_index(drop=True)],
        axis=1,
    )
    test_df = pd.concat(
        [test_gene_df.reset_index(drop=True), test_cell_df.reset_index(drop=True)],
        axis=1,
    )
    # Feature selection with variance encoding
    selector = VarianceThreshold(THRESHOLD)
    selector.fit_transform(train_df)
    train_df = train_df[train_df.columns[selector.get_support(indices=True)]]
    selected_cols = train_df.columns.tolist()
    test_df = test_df[selected_cols]

    # Log the shapes of train and test sets before and after processing
    logging.info("Data processing completed:")
    logging.info(f"Train features shape (before processing): {train_features_df.shape}")
    logging.info(f"Test features shape (before processing): {test_features_df.shape}")
    logging.info(f"Train features shape (after processing): {train_df.shape}")
    logging.info(f"Test features shape (after processing): {test_df.shape}")

    # Save processed data
    train_df.to_csv(output_path / "train_features.csv", index=False)
    test_df.to_csv(output_path / "test_features.csv", index=False)
    train_targets_df.to_csv(output_path / "train_targets.csv", index=False)
    test_targets_df.to_csv(output_path / "test_targets.csv", index=False)

    # Estimate time and memory consumption
    elapsed_time = time.time() - start_time
    process_elapsed_time = time.process_time() - process_start_time
    memory_usage = psutil.Process().memory_info().rss / 1024**2

    # Log the elapsed time and memory usage
    logging.info("Processed files have been saved in the data/processed directory.")
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info(f"Process elapsed time: {process_elapsed_time:.2f} seconds")
    logging.info(f"Memory usage: {memory_usage:.2f} MB")


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path: str, output_path: str) -> None:
    """
    Main function to handle data processing.

    Args:
        input_path (str): Path to the input data directory as a string.
        output_path (str): Path to the output data directory as a string.
    """
    # Seed everything for reproducibility
    seed_everything(SEED)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_path = Path(input_path)
    output_path = Path(output_path)

    process_data(input_path, output_path)


if __name__ == "__main__":
    main()
