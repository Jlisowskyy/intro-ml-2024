"""
Author: Tomasz Mycielski

Module featuring training functions plus a training setup
"""
from datetime import datetime
from random import randint

from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm  # for the progress bar

from src.cnn.cnn import BasicCNN
from src.cnn.loadset import MultiLabelDataset
from src.cnn.validator import Validator
from src.constants import TRAINING_TRAIN_BATCH_SIZE, TRAINING_TEST_BATCH_SIZE, \
    TRAINING_EPOCHS, TRAINING_LEARNING_RATES, TRAINING_VALIDATION_SET_SIZE, \
    TRAINING_TRAIN_SET_SIZE, TRAINING_TEST_SET_SIZE, TRAINING_MOMENTUM, DATABASE_ANNOTATIONS_PATH, \
    DATABASE_OUT_PATH, TRAINING_VALIDATION_BATCH_SIZE


def train_single_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optim: Optimizer,
        device: str,
        calculate_accuracy: bool = False,
        labels: LabelEncoder | None = None
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

    calculate_accuracy:
        # TODO: add description
    """

    validator = Validator(labels)
    train_loss = 0.0
    for input_data, target in tqdm(data_loader, colour='blue'):
        input_data, target = input_data.to(device), target.to(device)
        # calculate loss
        predictions = model(input_data)
        loss = loss_fn(predictions, target)
        if calculate_accuracy:
            validator.validate(predictions, target) # TODO: decouple this
        # back propagate error and update weights
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()

    print(f"Training loss: {train_loss / len(data_loader)}")
    if calculate_accuracy:
        validator.display_results()


def validate(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        device: str = 'cpu'
) -> float:
    """
    Function for evaluating `model` against an independent dataset during training

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
    valid_loss = 0.0
    model.eval()
    for input_data, target in tqdm(data_loader, colour='yellow'):
        input_data, target = input_data.to(device), target.to(device)
        predictions = model(input_data)
        loss = loss_fn(predictions, target)
        valid_loss += loss.item()
    model.train()
    return valid_loss


def train(model: nn.Module, train_data: DataLoader, loss_fn: nn.Module, optim: Optimizer,
          device: str, epochs: int, val_data: DataLoader | None = None,
          labels: LabelEncoder | None = None) -> None:
    """
    Method training `model` a set amount of epochs, outputting loss every iteration

    Parameters
    ----------
    model: :class:`torch.nn.Module`
        Model to train
    
    train_data: :class:`torch.utils.data.DataLoader`
        Dataloader to feed the model

    loss_fn: :class:`torch.nn.Module`
        Loss criterion
        
    optim: :class:`torch.optim.optimizer.Optimizer`
        Optimization criterion

    device: :class:`str`
        Can be either 'cuda' or 'cpu', set device for pytorch

    epochs: :class:`int`
        set amount of epochs to train the model

    val_data: :class:`torch.utils.data.DataLoader`
        # TODO: add description
    """
    min_valid_loss = float('inf')
    for i in range(1):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, train_data, loss_fn, optim, device, i == epochs - 1, labels)

        if val_data is None:
            continue

        valid_loss = validate(model, val_data, loss_fn, device)
        print(f'Validation loss: {valid_loss / len(val_data)}')
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            # backup for longer training sessions
            torch.save(model.state_dict(), f'cnn_e{i + 1}_backup.pth')
    print("Finished training")


def test(model: nn.Module, data_loader: DataLoader, device: str = 'cpu',
         labels: LabelEncoder | None = None) -> Validator:
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

    validator = Validator(labels)
    model.eval()
    with torch.no_grad():
        for input_data, target in tqdm(data_loader, colour='green'):
            input_data = input_data.to(device)
            predictions = model(input_data)
            validator.validate(predictions, target)

    validator.display_results()
    model.train()
    return validator


def main() -> None:
    """
    Main function for training the CNN model
    """

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device}')

    # preparing datasets
    dataset = MultiLabelDataset(
        DATABASE_ANNOTATIONS_PATH,
        DATABASE_OUT_PATH,
        device
    )

    seed = randint(0, 1 << 64)
    print(f'split seed: {seed}')
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = (
        torch.utils.data.random_split(dataset,
                                      [TRAINING_TRAIN_SET_SIZE, TRAINING_VALIDATION_SET_SIZE,
                                       TRAINING_TEST_SET_SIZE], generator=generator))

    train_dataloader = DataLoader(train_dataset, batch_size=TRAINING_TRAIN_BATCH_SIZE)
    validate_dataloader = DataLoader(validation_dataset, batch_size=TRAINING_VALIDATION_BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=TRAINING_TEST_BATCH_SIZE)

    # training
    for _, learning_rate in enumerate(TRAINING_LEARNING_RATES):
        cnn = BasicCNN(len(dataset.get_labels())).to(device)
        print(cnn)

        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=TRAINING_MOMENTUM)

        train(cnn, train_dataloader, loss_function, optimiser, device, TRAINING_EPOCHS,
              validate_dataloader, dataset.get_encoder())

        now = datetime.now().strftime('%Y-%m-%dT%H:%M')
        torch.save(cnn.state_dict(), f'cnn_{seed}_{now}.pth')
        test(cnn, test_dataloader, device, dataset.get_encoder())
