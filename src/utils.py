import numpy as np
import random
import os
import torch

import zipfile


def unzip_data(zip_file_path, extract_to_dir):
    """Extract zip data to the choosen folder

    Args:
        zip_file_path ('str'): path to zip file
        extract_to_dir ('str'): directory extract to
    """
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_dir)


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


# input_folder = "data/raw"
# output_folder = "data/processed"
# zip_file_path = input_folder + "/lish-moa-dataset.zip"

# unzip_data(zip_file_path, "data/to_check")