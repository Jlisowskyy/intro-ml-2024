"""
Author: Tomasz Mycielski
Implementation of the CNN
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    cnn = BasicCNN().to('cuda')
    summary(cnn, (3, 300, 400))
