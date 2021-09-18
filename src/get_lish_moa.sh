#!/bin/bash
export KAGGLE_CONFIG_DIR=/home/os/miniconda3/envs/moa/bin
kaggle competitions download -c lish-moa
unzip lish-moa.zip -d /home/os/Projects/moa/data/raw
rm *.zip