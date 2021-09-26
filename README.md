# Mechanisms of Action (MoA) Prediction

_The main goal of this research project is to develop an efficient algorithm for classifying drugs based on their biological activity._

## :book: About

The [Connectivity Map](https://clue.io/), a project within the Broad Institute of MIT and Harvard, the [Laboratory for Innovation Science at Harvard (LISH)](https://lish.harvard.edu/), and the [NIH Common Funds Library of Integrated Network-Based Cellular Signatures (LINCS)](https://lincsproject.org/), present this challenge with the goal of advancing drug development through improvements to MoA prediction algorithms.[(1)](#link-references)

> What is the Mechanism of Action (MoA) of a drug? And why is it important?

In pharmacology, the term mechanism of action (MOA) refers to the specific biochemical interaction through which a drug substance produces its pharmacological effect.[(2)](#link-references) A mechanism of action usually includes mention of the specific molecular targets to which the drug binds, such as an enzyme or receptor.[(3)](#link-references)

In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.

> How do we determine the MoAs of a new drug?

One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression ([GEO](https://www.ncbi.nlm.nih.gov/geo/), [EMBL-EBI Expression Atlas](https://www.ebi.ac.uk/gxa/home), etc.) or cell viability patterns of drugs with known MoAs.

<div align="center">
  <img src="reports/figures/mao_aim.png"/>
</div>

## :chart_with_upwards_trend: Evaluation Metric

Based on the MoA annotations, the accuracy of solutions will be evaluated on the average value of the [logarithmic loss function](https://www.kaggle.com/c/lish-moa/overview/evaluation) applied to each drug-MoA annotation pair.

<div align="center">
<img style="background: white;" src="https://latex.codecogs.com/gif.latex?\text{score}=-\frac{1}{M}\sum_{m=1}^{M}\frac{1}{N}\sum_{i=1}^{N}\left[y_{i,m}\log(\hat{y}_{i,m})+(1-y_{i,m})\log(1-\hat{y}_{i,m})\right]"/>
</div>


## :floppy_disk: Dataset

In this challenge, we have an access to a unique dataset that combines gene expression and cell viability data. The data is based on a new technology that measures simultaneously (within the same samples) human cells’ responses to drugs in a pool of 100 different cell types (thus solving the problem of identifying ex-ante, which cell types are better suited for a given drug). In addition, we have access to MoA annotations for more than 5,000 drugs in this dataset.

The training data has an additional (optional) set of MoA labels that are not included in the test data and not used for scoring.

In this competition, we need to predict multiple targets of the Mechanism of Action (MoA) response(s) of different samples (sig_id), given various inputs such as gene expression data and cell viability data.

<div align="center">
  <img src="reports/figures/gene_dist.png"/>
  <img src="reports/figures/cell_dist.png"/>
</div>

**List of files:**

- `train_features.csv` - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low).
- `train_drug.csv` - This file contains an anonymous drug_id for the training set only.
- `train_targets_scored.csv` - The binary MoA targets that are scored.
- `train_targets_nonscored.csv` - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
- `test_features.csv` - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
- `sample_submission.csv` - A submission file in the correct format.

## Data Preprocessing Pipeline

> 📌 **Note:** The project automation workflow is based on [Make GNU](https://en.wikipedia.org/wiki/Make_(software)) and integrated with the Kaggle API. The data preprocessing pipeline is part of the project automation workflow. 

To build a preprocessing pipeline, run the command `make data` in your terminal. This command triggers a chain of scripts in the following order:

1. Check the `data/raw` directory to make sure there is a training dataset.
2. Download dataset from Kaggle server if `data/raw` is empty, or skip this step otherwise.
3. Extract downloaded dataset to the `data/raw` directory
4. Delete downloaded zip file
5. Perform feature engineering tasks to prepare dataset for training
6. Perform feature selection
7. Save the prepared dataset in the `data/processed` directory
  
To make this possible, you must first [configure your Kaggle API credentials.](#closed_lock_with_key-how-to-use-the-kaggle-api). 

> 📌 **Note:** If you do not want to use Kaggle CLI tools you can [download and extract the dataset manually](#inbox_tray-how-to-download-and-extract-the-dataset-manually). It will not harm the data preprocessing pipeline.

## :inbox_tray: How to Download and Extract the Dataset Manually

Follow the next steps to access and download data manually:

1. Sign in to your [Kaggle](https://www.kaggle.com/) account or sign up if you haven't an account yet.
2. Accept [MoA competition rules](https://www.kaggle.com/c/lish-moa/rules) - it will grant you full access to MoA competition data.
3. Download the [dataset](https://www.kaggle.com/c/lish-moa/data).
4. Unzip downloaded `lish_moa.zip` file to the `data/raw` project directory.

 ## :rocket: Models

We use PyTorch as a primary deep learning framework for this project. The current solution is based on two architectures, the first is one-dimensional CNN and the second is PyTorch TabNet. Both architectures are adapted for the task of multi-label classification and fine-tuned for better performance.

#### TabNet

<div align="center">
  <img src="reports/figures/tabnet.png"/>
</div>

## :gear: The Project Automation Workflow

As was previously mentioned, project automation workflow is based on [Make GNU](https://www.gnu.org/software/make/). The core of this The Makefile is a core of this workflow - is are just rules that form a chain of a high abstraction level and connect all the processes inside the project together. 

#### All `make` commands:

| Command                 | Description                      | Prerequisite       |
| ----------------------- | -------------------------------- | ------------------ |
| `make environment`      | Create a virtual environment     |                    |
| `source moa activate`   | Activate virtual environment     |                    |
| `make test_environment` | Test virtual environment         |                    |
| `make requirements`     | Install dependencies             | `test_environment` |
| `make get_data`         | Download and extract data        |                    |
| `make data`             | Make data preprocessing pipeline | `get_data`         |
| `make train`            | Initialize model training        | `data`             |
| `make prediction `      | Make prediction                  | `train`            |
| `make report`           | Create report                    |                    |
| `make clean`            | Delete all compiled Python files |                    |
| `make lint`             | Lint using flake8                |                    |

**What Does the `Prerequisite` Column Mean?**

The entries in the `Prerequisite` column indicate that a particular command is based on the top of a particular condition. For example, `data` is a prerequisite for the `make train` command. The logic here is pretty simple - you need to prepare the data before starting training. But you don't need to run `prerequisites` manually. When you run the target command in the terminal, it will run the prerequisite automatically and continue only if successful.

## :closed_lock_with_key: How to Use the Kaggle API

Follow these steps to set up the Kaggle API credentials:

1. Create a new Kaggle API token, according to the [instructions](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).
2. Save obtained `kaggle.json` file to the `~/.kaggle` folder.
   
> 📌 **Note:** If you need to store the Kaggle API token in an environment location, you must set the **KAGGLE_CONFIG_DIR** environment variable to the path where you store the Kaggle API token **kaggle.json**. For example, on a Unix-based machine, the command would look like this:

```bash
    export KAGGLE_CONFIG_DIR=/home/user/miniconda3/envs/moa/bin
```

For your security, ensure that other users of your computer do not read access to your credentials: `chmod 600 ~/.kaggle/kaggle.json`

You can also choose to export your Kaggle username and token to the environment:

```bash
    export KAGGLE_USERNAME=niander_wallace
    export KAGGLE_KEY=xxxxxxxxxxxxxx
```

Follow the [documentation](https://www.kaggle.com/docs/api) to learn more about the Kaggle API and how to use Kaggle CLI tools.

## :open_file_folder: Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── predictions    <- Predicted targets
    │   ├── processed      <- The final, canonical data sets for modeling. Obtained after
    │   │                     preprocessing, merging, cleaning, feature engineering etc.
    │   └── raw            <- The original, immutable data dump. Should be considered as read only.
    │
    ├── drafts             <- Drafts, hypothesis testing
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                     the creator's initials, and a short `-` delimited description, e.g.
    │   │                     `1.0-os-initial-data-exploration`
    │   ├── exploratory    <- Contains initial explorations
    │   └── reports        <- Works that can be exported as html to the reports directory
    │
    ├── notes              <- Notes, ideas, experiment tracking, etc.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project
    │
    ├── test_environment   <- Test python environment is setup correctly
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## :bulb: How to Reproduce the Solution

Follow the steps bellow to reproduce the solution:

1. Clone the repository: `git clone https://github.com/oleksandrsirenko/mechanisms-of-action-moa-prediction.git moa`
2. [Get dataset manually](#inbox_tray-how-to-get-data) or [configure the Kaggle API](#closed_lock_with_key-how-to-use-the-kaggle-api) to automate this process.
3. Create a virtual environment for the project: `make environment`
4. Activate virtual environment: `source moa activate`
5. Install dependencies: `make requirements`
6. Prepare dataset: `make data`
7. Train models: `make train`
8. Make predictions: `make prediction`

## :link: References

 1. [Mechanisms of Action (MoA) Prediction](https://www.kaggle.com/c/lish-moa).
 2. Spratto, G.R.; Woods, A.L. (2010). Delmar Nurse's Drug Handbook. Cengage Learning. ISBN 978-1-4390-5616-5.
 3. Grant, R.L.; Combs, A.B.; Acosta, D. (2010) "Experimental Models for the Investigation of Toxicological Mechanisms". In McQueen, C.A. Comprehensive Toxicology (2nd ed.). Oxford: Elsevier. p. 204. ISBN 978-0-08-046884-6.
 4. Corsello et al. [“Discovering the anticancer potential of non-oncology drugs by systematic viability profiling”](https://doi.org/10.1038/s43018-019-0018-6), Nature Cancer, 2020.
 5. [GEO](https://www.ncbi.nlm.nih.gov/geo/) is a public functional genomics data repository supporting MIAME-compliant data submissions.
 6. [EMBL-EBI Expression Atlas](https://www.ebi.ac.uk/gxa/home)
 7. Subramanian et al. [“A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles”](https://doi.org/10.1016/j.cell.2017.10.049), Cell, 2017.
 8. [Connectopedia](https://clue.io/connectopedia/glossary) is a free, web-based dictionary of terms and concepts related to the Connectivity Map (including definitions of cell viability and gene expression data in that context.

## Current Status

> In progress

TODO list:

- [x] Define project structure
- [x] Automate workflow with Makefile
- [x] Integrate Kaggle API
- [x] Create data preprocessing pipeline
- [x] Make Dataset class
- [x] Make metrics
- [x] Define helper functions
- [x] Build baseline PyTorch model
- [ ] Construct training loop
- [ ] Create TabNet model
- [ ] Ensemble models
- [ ] Make inference
- [ ] Automate report fetching
- [ ] Implement remote training on GPU