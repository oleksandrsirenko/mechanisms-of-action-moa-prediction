.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = moa
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install python dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Download dataset if not exist
get_gata:
ifneq ("$(wildcard $(PROJECT_DIR)/data/raw/train_drug.csv)", "")
	@echo ">>> MOA dataset already exists and is ready for preprocessing."
else
	@echo ">>> Data folder is empty. Downloading ..."
	kaggle competitions download -c lish-moa
	unzip lish-moa.zip -d $(PROJECT_DIR)/data/raw
	rm *.zip
	@echo ">>> MOA dataset is ready for preprocessing."
endif

## Proprocess dataset
data: get_gata
	$(PYTHON_INTERPRETER) src/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python=3
	@echo ">>> New conda env created. Activate with:\
	\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m venv $(PROJECT_NAME)
	@echo ">>> New python env created. Activate on Unix or MacOS with:\
	\nsource $(PROJECT_NAME)/bin/activate"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py