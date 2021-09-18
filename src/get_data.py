# This script will automatically download the dataset using Kaggle API

# TODO: Make sure the following steps have done before execute the script:

# 1. Created virtual environment for the project
# 2. Activated virtual environment
# 3. Installed all requrements including Kaggle with: `make requirements`
# 4. Created Kaggle API Token saved to the folder `~/.kaggle/kaggle.json`
#    or to location of you choice eg. home/user/conda/env/moa/bin
# 5. Configurate lish_moa.sh file:
#        - set KAGGLE_CONFIG_DIR equal to folder with saved kaggle.json
#        - set path to the data/raw to unzip the data
# 6. Optional: `chmod 600 <path to kaggle.json>` to set permissions

# details: https://github.com/Kaggle/kaggle-api#api-credentials

# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
import subprocess


@click.command()
@click.argument("raw_data_path", type=click.Path(exists=True))
@click.argument("bash_script_path", type=click.Path())
def main(raw_data_path: Path, bash_script_path: Path) -> None:
    """Download end extract data if data/raw folder is empty
    Args:
        raw_data_path (Path): path to raw data
        bash_script_path (Path): path to bash script to download data
    """
    logger = logging.getLogger(__name__)
    logger.info("Check if raw data exists")

    train_path = os.path.join(raw_data_path, "train_drug.csv")
    if not os.path.isfile(train_path):
        logger.info("Data folder is empty. Downloading ...")
        subprocess.call(bash_script_path)
        logger.info("MOA dataset has been downloaded and is ready for preprocessing.")
    else:
        logger.info("MOA dataset already exists and is ready for preprocessing.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
