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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    os.chdir(project_root)

    print("Current working directory:", os.getcwd())
    print("PyTorch version:", torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    return device

