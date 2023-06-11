import numpy as np
import random
import os
import torch
import json


def seed_everything(seed_value):
    """Set seed for reproducibility

    Args:
        seed_value ('int'): Numerical value of the seed
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        if (
            torch.backends.cudnn.version() != None
            and torch.backends.cudnn.enabled == True
        ):
            if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 6:
                torch.backends.cudnn.deterministic = True
            else:
                torch.backends.cudnn.benchmark = True


def load_config(config_path: str) -> dict:
    """Load the model configuration from a JSON file
    Args:
        config_path (str): Path to the configuration file
    Returns:
        dict: Loaded configuration
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
