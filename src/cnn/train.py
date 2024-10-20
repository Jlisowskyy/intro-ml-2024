"""
Author: Tomasz Mycielski

Module featuring training functions plus a training setup
"""
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm  # for the progress bar

from .cnn import TutorialCNN
from .loadset import DAPSDataset

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATES = [0.001]
SAMPLE_RATE = 16000
SAMPLE_COUNT = 16000 * 5


def train_single_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optim: Optimizer,
        device: str
) -> None:
    """Method training `model` a single iteration with the data provided

    Parameters
    ----------
    model: :class:`torch.nn.Module`
        Model to train
    
    data_loader: :class:`torch.utils.data.DataLoader`
        Dataloader to feed the model

    loss_fn: :class:`torch.nn.Module`
        Loss criterion
        
    optim: :class:`torch.optim.optimizer.Optimizer`
        Optimization criterion

    device: :class:`str`
        Can be either 'cuda' or 'cpu', set device for pytorch
    """

    loss = None
    for input_data, target in tqdm(data_loader):
        input_data, target = input_data.to(device), target.to(device)

        # calculate loss
        prediction = model(input_data)
        loss = loss_fn(prediction, target)

        # back propagate error and update weights
        optim.zero_grad()
        loss.backward()
        optim.step()

    if loss is not None:
        print(f"loss: {loss.item()}")


def train(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optim: Optimizer,
          device: str,
          epochs: int) -> None:
    """
    Method training `model` a set amount of epochs, outputting loss every iteration

    Parameters
    ----------
    model: :class:`torch.nn.Module`
        Model to train
    
    data_loader: :class:`torch.utils.data.DataLoader`
        Dataloader to feed the model

    loss_fn: :class:`torch.nn.Module`
        Loss criterion
        
    optim: :class:`torch.optim.optimizer.Optimizer`
        Optimization criterion

    device: :class:`str`
        Can be either 'cuda' or 'cpu', set device for pytorch

    epochs: :class:`int`
        set amount of epochs to train the model
    """

    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optim, device)
    print("Finished training")


def validate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: str = 'cpu'):
    """
    Validate `model`

    Parameters
    ----------
    model: :class:`torch.nn.Module`
        Model to train
    
    data_loader: :class:`torch.utils.data.DataLoader`
        Dataloader to feed the model

    loss_fn: :class:`torch.nn.Module`
        Loss criterion

    device: :class:`str`
        Can be either 'cuda' or 'cpu', set device for pytorch
    """
    for input_data, target in tqdm(data_loader):
        input_data, target = input_data.to(device), target.to(device)
        prediction = model(input_data)
        loss = loss_fn(prediction, target)
        print(prediction, target)
        loss.backward()
    print(loss.item())


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using {DEVICE}")

    dataset = DAPSDataset(
        './annotations.csv',
        './datasets/daps_split/',
        SAMPLE_RATE,
        SAMPLE_COUNT,
        DEVICE
    )

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for index, learning_rate in enumerate(LEARNING_RATES):
        cnn = TutorialCNN().to(DEVICE)
        print(cnn)

        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)

        train(cnn, train_dataloader, loss_function, optimiser, DEVICE, EPOCHS)
        torch.save(cnn.state_dict(), f'cnn{index:0>2}.pth')

        validate(cnn, test_dataloader, loss_function, 'cuda')
