# Mechanisms of Action (MoA) Prediction
_The main goal of this research project is to develop an efficient algorithm for classifying drugs based on their biological activity._

## Description
---
The [Connectivity Map](https://clue.io/), a project within the Broad Institute of MIT and Harvard, the [Laboratory for Innovation Science at Harvard (LISH)](https://lish.harvard.edu/), and the [NIH Common Funds Library of Integrated Network-Based Cellular Signatures (LINCS)](https://lincsproject.org/), present this challenge with the goal of advancing drug development through improvements to MoA prediction algorithms.[[1]](#1)

>What is the Mechanism of Action (MoA) of a drug? And why is it important?

In pharmacology, the term mechanism of action (MOA) refers to the specific biochemical interaction through which a drug substance produces its pharmacological effect.[[2]](#2) A mechanism of action usually includes mention of the specific molecular targets to which the drug binds, such as an enzyme or receptor.[[3]](#3)

In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.

> How do we determine the MoAs of a new drug?

One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression ([GEO](https://www.ncbi.nlm.nih.gov/geo/), [EMBL-EBI Expression Atlas](https://www.ebi.ac.uk/gxa/home), etc.) or cell viability patterns of drugs with known MoAs.

As is customary, the dataset has been split into testing and training subsets. Hence, our task is to use the training dataset to develop an algorithm that automatically labels each case in the test set as one or more MoA classes. Note that since drugs can have multiple MoA annotations, the task is formally a multi-label classification problem.

## Evaluation
---
Based on the MoA annotations, the accuracy of solutions will be evaluated on the average value of the [logarithmic loss function](https://www.kaggle.com/c/lish-moa/overview/evaluation) applied to each drug-MoA annotation pair.

## Data
---
In this challenge, we have an access to a unique dataset that combines gene expression and cell viability data. The data is based on a new technology that measures simultaneously (within the same samples) human cells’ responses to drugs in a pool of 100 different cell types (thus solving the problem of identifying ex-ante, which cell types are better suited for a given drug). In addition, we have access to MoA annotations for more than 5,000 drugs in this dataset.

The training data has an additional (optional) set of MoA labels that are not included in the test data and not used for scoring.

In this competition, we need to predict multiple targets of the Mechanism of Action (MoA) response(s) of different samples (sig_id), given various inputs such as gene expression data and cell viability data.

**Note:** by default, the `data` folder is included in the `.gitignore` file. To reproduce the solution you need to download the original dataset from [the official web site of the competition](https://www.kaggle.com/c/lish-moa) and extract a zip archive in the `data/row` folder. 

Files:

- `train_features.csv` - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low).
- `train_drug.csv` - This file contains an anonymous drug_id for the training set only.
- `train_targets_scored.csv` - The binary MoA targets that are scored.
- `train_targets_nonscored.csv` - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
- `test_features.csv` - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
- `sample_submission.csv` - A submission file in the correct format.

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                     the creator's initials, and a short `-` delimited description, e.g.
    │   │                     `1.0-jqp-initial-data-exploration`
    │   ├── exploratory    <- Contains initial explorations
    │   └── reports        <- Works that can be exported as html to the reports directory
    │
    ├── prototyping        <- Experiments with data, engineering approaches, models, etc.
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
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

### Installing development requirements
---

    pip install -r requirements.txt

### Running the tests
---

    py.test tests
---

## References
---
 <a name="1">1.</a> [Kaggle](https://www.kaggle.com/) is the biggest data science community and the official hoster of the [compete](https://www.kaggle.com/c/lish-moa).
 
 <a name="2">2.</a> Spratto, G.R.; Woods, A.L. (2010). Delmar Nurse's Drug Handbook. Cengage Learning. ISBN 978-1-4390-5616-5.

 <a name="3">3.</a> Grant, R.L.; Combs, A.B.; Acosta, D. (2010) "Experimental Models for the Investigation of Toxicological Mechanisms". In McQueen, C.A. Comprehensive Toxicology (2nd ed.). Oxford: Elsevier. p. 204. ISBN 978-0-08-046884-6.

<a name="4">4.</a> Corsello et al. [“Discovering the anticancer potential of non-oncology drugs by systematic viability profiling”](https://doi.org/10.1038/s43018-019-0018-6), Nature Cancer, 2020.

<a name="5">5.</a> Subramanian et al. [“A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles”](https://doi.org/10.1016/j.cell.2017.10.049), Cell, 2017.

<a name="6">6.</a> [Connectopedia](https://clue.io/connectopedia/glossary) is a free, web-based dictionary of terms and concepts related to the Connectivity Map (including definitions of cell viability and gene expression data in that context).