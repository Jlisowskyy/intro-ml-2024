"""
Implementation of the CNN
"""
from torch import nn
from torchsummary import summary


class CNN(nn.Module):
    """
    Simple CNN with 4 convolution layers
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        # note the tensor size changes when data shape changes
        self.linear = nn.Linear(7040, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        """
        Data processing method
        """
        data = self.conv1(input_data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.flatten(data)
        logits = self.linear(data)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNN()
    summary(cnn.cuda(), (1, 64, 157))
