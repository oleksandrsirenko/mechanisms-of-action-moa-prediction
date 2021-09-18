#!/bin/bash
export KAGGLE_CONFIG_DIR=<path to the Kaggle API token kaggle.json file>
kaggle competitions download -c lish-moa
unzip lish-moa.zip -d <full path to data/raw folder>
rm *.zip