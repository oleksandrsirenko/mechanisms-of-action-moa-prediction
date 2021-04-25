import numpy as np
import pandas as pd
import random
import os
import sys
import zipfile

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings

warnings.filterwarnings("ignore")

SEED = 42


def seed_everything(seed_value):
    """Set seed to reprodusability

    Args:
        seed_value ('int'): numerical value of the seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unzip_data(zip_file_path, extract_to_dir):
    """Extract zip data to the choosen folder

    Args:
        zip_file_path ('str'): path to zip file
        extract_to_dir ('str'): directory extract to
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_dir)


class PreprocessRawData:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.train = pd.read_csv(f"{input_dir}/train_features.csv")
        self.test = pd.read_csv(f"{input_dir}/test_features.csv")


data_list = os.listdir("data/raw")
print(data_list)

"""def main():

    input_dir = sys.argv[1].lower()
    zip_file_path = f"{input_dir}/lish-moa-dataset.zip"

    output_dir = sys.argv[2].lower()

    # extract data
    unzip_data(zip_file_path, input_dir)

    # set seed
    seed_everything(seed_value=SEED)

    # preprocess and save data
    process = PreprocessRawData(input_dir, output_dir)
    process.make_features()
    process.select_features()
    process.save()


if __name__ == "__main__":
    main()"""
