from typing import List, Tuple, Dict
import numpy as np
import random
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import log_loss

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


def make_transform_pipe(
    n_pca: int = 600, n_qt: int = 100, n_kbd: int = 10
) -> FeatureUnion:

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
    transforms.append(("svd", TruncatedSVD(n_components=n_pca)))

    return FeatureUnion(transforms)


def make_gaussian_trans(
    features: Dict[str, List[str]], train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame]:

    transformer = QuantileTransformer(
        n_quantiles=100, random_state=42, output_distribution="normal"
    )
    for col in features:
        train_vec_len = len(train[col].values)
        test_vec_len = len(test[col].values)
        raw_vec = train[col].values.reshape(train_vec_len, 1)

        transformer.fit(raw_vec)

        train[col] = transformer.transform(raw_vec).reshape(1, train_vec_len)[0]
        test[col] = transformer.transform(
            test[col].values.reshape(test_vec_len, 1)
        ).reshape(1, test_vec_len)[0]

    return train, test


class TransformFeatures:
    def __init__(self, train, test) -> None:
        self.train = train
        self.test = test


def make_pca_trans(
    n_comps: int,
    flag: str,
    seed: int,
    features: Dict[str, List[str]],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame]:
    cols = features[flag]
    data = pd.concat(train[cols], test[cols])
    data = PCA(n_components=n_comps, random_state=seed).fit_trainsform(data[cols])

    return train, test


def select_features():
    pass
