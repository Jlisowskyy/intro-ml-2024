"""
Author: Tomasz Mycielski

Module featuring training functions plus a training setup
"""
from datetime import datetime

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm  # for the progress bar

from src.cnn.cnn import BasicCNN
from src.cnn.loadset import DAPSDataset
from src.constants import TRAINING_TEST_BATCH_SIZE, TRAINING_VALIDATION_BATCH_SIZE, \
    TRAINING_EPOCHS, TRAINING_LEARNING_RATES, \
    TRAINING_TRAIN_SET_SIZE, TRAINING_TEST_SET_SIZE, TRAINING_MOMENTUM, DATABASE_ANNOTATIONS_PATH, \
    DATABASE_OUT_PATH
from src.validation.simple_validation import SimpleValidation


def train_single_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optim: Optimizer,
        device: str,
        calculate_accuracy: bool = False
) -> None:
    """
    Method training `model` a single iteration with the data provided

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

    validator = SimpleValidation()
    loss = None
    for input_data, target in tqdm(data_loader, colour='blue'):
        input_data, target = input_data.to(device), target.to(device)

        # calculate loss
        predictions = model(input_data)
        loss = loss_fn(predictions, target)
        if calculate_accuracy:
            validator.validate(predictions, target)
        # back propagate error and update weights
        optim.zero_grad()
        loss.backward()
        optim.step()

    if loss is not None:
        print(f"loss: {loss.item()}")
    if calculate_accuracy:
        validator.display_results()



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
        train_single_epoch(model, data_loader, loss_fn, optim, device, i == epochs - 1)
        # # backup for longer training sessions
        # torch.save(model.state_dict(), f'model_epoch_{i+1}_backup.pth')
    print("Finished training")


def validate(model: nn.Module, data_loader: DataLoader, device: str = 'cpu'):
    """
    Validates binary classification `model`
    Prints results including TP/FP/FN/TN, accuracy and F1 score to stdout

    Parameters
    ----------
    model: :class:`torch.nn.Module`
        Model to train
    
    data_loader: :class:`torch.utils.data.DataLoader`
        Dataloader to feed the model

    device: :class:`str`
        Can be either 'cuda' or 'cpu', set device for pytorch
    """

    validator = SimpleValidation()
    model.eval()
    with torch.no_grad():
        for input_data, target in tqdm(data_loader, colour='green'):
            input_data = input_data.to(device)
            predictions = model(input_data)
            validator.validate(predictions, target)

    validator.display_results()
    model.train()


if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    print(f'Using {DEVICE}')

    dataset = DAPSDataset(
        DATABASE_ANNOTATIONS_PATH,
        DATABASE_OUT_PATH,
        DEVICE
    )

    train_dataset, test_dataset = (
        torch.utils.data.random_split(dataset, [TRAINING_TRAIN_SET_SIZE,
                                                TRAINING_TEST_SET_SIZE]))

    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_TEST_BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=TRAINING_VALIDATION_BATCH_SIZE)

    for index, learning_rate in enumerate(TRAINING_LEARNING_RATES):

        cnn = BasicCNN().to(DEVICE)
        print(cnn)

        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=TRAINING_MOMENTUM)

        train(cnn, train_dataloader, loss_function, optimiser, DEVICE, TRAINING_EPOCHS)

        now = datetime.now().strftime('%Y-%m-%dT%H:%M')
        torch.save(cnn.state_dict(), f'cnn_{now}.pth')
        validate(cnn, test_dataloader, DEVICE)
