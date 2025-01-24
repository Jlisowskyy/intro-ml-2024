"""
This is a helper script to setup the notebook environment.
"""

import os
import torch


def setup_notebook_env():
    """
    Setup the notebook environment
    Returns: device
    """
    os.chdir("../")
    print("Current working directory:", os.getcwd())
    print("PyTorch version:", torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    return device
