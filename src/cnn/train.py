"""
Module featuring training functions plus a training setup
"""
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from .cnn import CNN
from .loadset import DAPSDataset

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001
SAMPLE_RATE = 16000
SAMPLE_COUNT = 16000 * 5


def train_single_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optim: Optimizer,
        device: str
    ):
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
    for input_data, target in data_loader:
        input_data, target = input_data.to(device), target.to(device)

        # calculate loss
        prediction = model(input_data)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optim, device, epochs):
    """Method training `model` a set amount of epochs, outputting loss every iteration

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


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using {DEVICE}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = DAPSDataset(
        './datasets/annotations.csv',
        './datasets/daps',
        mel_spectrogram,
        SAMPLE_RATE,
        SAMPLE_COUNT,
        DEVICE
    )

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    cnn = CNN().to(DEVICE)
    print(cnn)

    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    train(cnn, train_dataloader, loss_function, optimiser, DEVICE, EPOCHS)
    torch.save(cnn.state_dict(), "cnn.pth")
