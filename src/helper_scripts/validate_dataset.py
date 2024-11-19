"""
Author: Tomasz Mycielski

Module for classifying datasets with pretrained models
"""
import torch
import torch.utils
from torch.utils.data import DataLoader

from src.cnn.cnn import BasicCNN
from src.cnn.loadset import DAPSDataset
from src.cnn.train import test
from src.constants import TRAINING_TEST_SET_SIZE, TRAINING_VALIDATION_SET_SIZE, \
    TRAINING_TRAIN_SET_SIZE, DATABASE_ANNOTATIONS_PATH, DATABASE_OUT_PATH


def main() -> None:
    """
    Script entry point
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn = BasicCNN()
    cnn.load_state_dict(torch.load('./models/cnn_e9_4460117552633912135_2024-10-30T07:55.pth',
                                   map_location=torch.device(device),
                                   weights_only=True))
    cnn.to(device)

    dataset = DAPSDataset(
        DATABASE_ANNOTATIONS_PATH,
        DATABASE_OUT_PATH,
        device
    )

    generator = torch.Generator().manual_seed(4460117552633912135)
    train_dataset, validation_dataset, test_dataset = (
        torch.utils.data.random_split(dataset,
                                      [TRAINING_TRAIN_SET_SIZE, TRAINING_VALIDATION_SET_SIZE,
                                       TRAINING_TEST_SET_SIZE], generator=generator))

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    validate_dataloader = DataLoader(validation_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    print('Training dataset')
    res = test(cnn, train_dataloader, device)
    print('Validation dataset')
    res += test(cnn, validate_dataloader, device)
    print('Testing dataset')
    res += test(cnn, test_dataloader, device)
    print('Full dataset')
    res.display_results()
