"""
Author: Tomasz Mycielski, 2024

Implementation of the CNN
"""
import torch
import torch.nn.functional as tnnf
from torch import nn


class BasicCNN(nn.Module):
    """
    Simplified CNN with two layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(111744, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 2)


    def forward(self, x):
        """
        Data processing method
        """
        x = self.pool(tnnf.relu(self.conv1(x)))
        x = self.pool(tnnf.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = tnnf.relu(self.fc1(x))
        x = tnnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x
