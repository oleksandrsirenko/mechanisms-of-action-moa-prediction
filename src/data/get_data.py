import numpy as np
import pandas as pd
import os
import random
from .utils import seed_everything

data_raw = "../input/lish-moa/"
os.listdir(data_dir)

train_features = pd.read_csv(data_dir + "train_features.csv")
train_targets_scored = pd.read_csv(data_dir + "train_targets_scored.csv")
train_targets_nonscored = pd.read_csv(data_dir + "train_targets_nonscored.csv")
train_drug = pd.read_csv(data_dir + "train_drug.csv")
test_features = pd.read_csv(data_dir + "test_features.csv")
sample_submission = pd.read_csv(data_dir + "sample_submission.csv")

print("train_features: {}".format(train_features.shape))
print("train_targets_scored: {}".format(train_targets_scored.shape))
print("train_targets_nonscored: {}".format(train_targets_nonscored.shape))
print("train_drug: {}".format(train_drug.shape))
print("test_features: {}".format(test_features.shape))
print("sample_submission: {}".format(sample_submission.shape))

###

for col in GENES + CELLS:
    transformer = QuantileTransformer(
        n_quantiles=100, random_state=0, output_distribution="normal"
    )
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(
        test_features[col].values.reshape(vec_len_test, 1)
    ).reshape(1, vec_len_test)[0]
