.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = moa
PYTHON_INTERPRETER = python3
MODEL = MoaModel

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment (conda or python venv)
env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create --name $(PROJECT_NAME) python=3
	@echo ">>> New conda env created. Activate with:\
	\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> New python env created. Activate on Unix or MacOS with:\
	\nsource $(PROJECT_NAME)/bin/activate"
endif

## Test python environment is set up correctly
test_env:
	$(PYTHON_INTERPRETER) test_environment.py

## Install python dependencies based on the environment
requirements: test_env
ifeq (True,$(HAS_CONDA))
	@echo ">>> Installing conda dependencies..."
	conda install --name $(PROJECT_NAME) --file requirements.txt
else
	@echo ">>> Installing pip dependencies..."
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
endif

## Download dataset if it doesn't exist
raw_data:
	mkdir -p $(PROJECT_DIR)/data/raw
ifeq ("$(wildcard $(PROJECT_DIR)/data/raw/train_drug.csv)", "")
	@echo ">>> Data folder is empty. Downloading ..."
	kaggle competitions download -c lish-moa
	unzip lish-moa.zip -d $(PROJECT_DIR)/data/raw
	rm *.zip
	@echo ">>> MOA dataset is ready for preprocessing."
else
	@echo ">>> MOA dataset already exists and is ready for preprocessing."
endif

## Preprocess dataset
data: raw_data
	mkdir -p $(PROJECT_DIR)/data/processed/
	$(PYTHON_INTERPRETER) src/dataset.py data/raw/ data/processed/

## Initialize main training loop
train:
ifeq ($(shell find $(PROJECT_DIR)/data/processed -name "*.csv" | wc -l),0)
	@echo "Processed files not found. Running preprocessing before training."
	$(MAKE) data
else
	@echo "Processed files already exist. Skipping preprocessing and proceeding to training."
	$(PYTHON_INTERPRETER) src/train.py --model_name $(MODEL) --config configs/$(MODEL)_config.json
endif


## Inference with ensembling pre-trained models
pred: train
	$(PYTHON_INTERPRETER) src/predict.py models/ data/processed/ data/predictions/

## Create report
report:
	$(PYTHON_INTERPRETER) make_report.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

# List all available `make` targets and their descriptions
help:
	@echo "Available targets:"
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = $$1; \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  \033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
