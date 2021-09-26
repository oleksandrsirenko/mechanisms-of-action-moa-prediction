# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd

# from utils import seed_everything

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA


SEED = 42
THRESHOLD = 0.9


def make_transform_pipe(n_pca: int, n_qt: int = 100, n_kbd: int = 10) -> FeatureUnion:

    transforms = list()
    transforms.append(("mms", MinMaxScaler()))
    transforms.append(("ss", StandardScaler()))
    transforms.append(("rs", RobustScaler()))
    transforms.append(
        ("qt", QuantileTransformer(n_quantiles=n_qt, output_distribution="normal"))
    )
    transforms.append(
        ("kbd", KBinsDiscretizer(n_bins=n_kbd, encode="ordinal", strategy="uniform"))
    )
    transforms.append(("pca", PCA(n_components=n_pca)))

    return FeatureUnion(transforms)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path: Path, output_path: Path) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Args:
        input_path (Path): data/raw
        output_path (Path): data/processed
    """
    logger = logging.getLogger(__name__)
    logger.info("Start making the final data set from raw data...")

    logger.info("Reading raw data...")
    train_features_df = pd.read_csv(input_path + "train_features.csv")
    train_targets_df = pd.read_csv(input_path + "train_targets_scored.csv")
    test_features_df = pd.read_csv(input_path + "test_features.csv")
    test_targets_df = pd.read_csv(input_path + "sample_submission.csv")

    # save targets without changing

    GENES = [col for col in train_features_df.columns if col.startswith("g-")]
    CELLS = [col for col in train_features_df.columns if col.startswith("c-")]

    # GENE FEATURES
    logger.info("Making gene feature transformation pipeline")
    gene_trans_pipe = make_transform_pipe(n_pca=600)

    logger.info("Fit-transform on gene training features")
    train_gene_features = gene_trans_pipe.fit_transform(train_features_df[GENES])
    TRANS_GENES = [f"g-{i}" for i in range(train_gene_features.shape[1])]
    train_gene_df = pd.DataFrame(train_gene_features, columns=TRANS_GENES)

    logger.info("Transform gene test features")
    test_gene_features = gene_trans_pipe.transform(test_features_df[GENES])
    test_gene_df = pd.DataFrame(test_gene_features, columns=TRANS_GENES)

    # CELL FEATURES
    logger.info("Making cell feature transdormation pipeline")
    cell_trans_pipe = make_transform_pipe(n_pca=50)

    logger.info("Fit-transform train cell features")
    train_cell_features = cell_trans_pipe.fit_transform(train_features_df[CELLS])
    TRANS_CELLS = [f"c-{i}" for i in range(train_cell_features.shape[1])]
    train_cell_df = pd.DataFrame(train_cell_features, columns=TRANS_CELLS)

    logger.info("Transform cell test features")
    test_cell_features = cell_trans_pipe.transform(test_features_df[CELLS])
    test_cell_df = pd.DataFrame(test_cell_features, columns=TRANS_CELLS)

    # CONCAT GENE AND CELL FEATURES
    logger.info("Concatenate gene and cell features after transformation")
    train_df = pd.concat([train_gene_df, train_cell_df], axis=1)
    test_df = pd.concat([test_gene_df, test_cell_df], axis=1)

    # FEATURE SELECTION WITH VARIANCE ENCODING
    selector = VarianceThreshold(THRESHOLD)
    logger.info(
        f"Select features using VarianceThreshold with threshold = {THRESHOLD} "
    )
    selector.fit_transform(train_df)
    train_df = train_df[train_df.columns[selector.get_support(indices=True)]]

    selected_cols = train_df.columns.tolist()
    test_df = test_df[selected_cols]
    logger.info("All transformations completed")
    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Test df shape: {test_df.shape}")

    # SAVE PREPROCESSED DATA TO outpath 'data/processed'
    logger.info("Saving processed data...")
    train_df.to_csv(output_path + "train_features.csv", index=False)
    test_df.to_csv(output_path + "test_features.csv", index=False)
    train_targets_df.to_csv(output_path + "train_targets.csv", index=False)
    test_targets_df.to_csv(output_path + "test_targets.csv", index=False)

    logger.info("Processed files have been saved in the data/processing directory.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # seed_everything(SEED)

    main()
