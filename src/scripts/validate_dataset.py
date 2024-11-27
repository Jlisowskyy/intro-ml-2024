"""
Author: Tomasz Mycielski

Module for classifying datasets with pretrained models
"""
import torch
import torch.utils
from torch.utils.data import DataLoader

from src.cnn.cnn import BaseCNN
from src.cnn.loadset import MultiLabelDataset
from src.cnn.train import test
from src.constants import (TRAINING_TEST_SET_SIZE, TRAINING_VALIDATION_SET_SIZE,
                           TRAINING_TRAIN_SET_SIZE, DATABASE_ANNOTATIONS_PATH,
                           DATABASE_OUT_PATH, MODEL_BASE_PATH)


def main() -> None:
    """
    Script entry point
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn = BaseCNN()
    cnn.load_state_dict(torch.load(MODEL_BASE_PATH,
                                   map_location=torch.device(device),
                                   weights_only=True))
    cnn.to(device)

    dataset = MultiLabelDataset(
        DATABASE_ANNOTATIONS_PATH,
        DATABASE_OUT_PATH,
        device
    )

    generator = torch.Generator().manual_seed(7363662207225070962)
    train_dataset, validation_dataset, test_dataset = (
        torch.utils.data.random_split(dataset,
                                      [TRAINING_TRAIN_SET_SIZE, TRAINING_VALIDATION_SET_SIZE,
                                       TRAINING_TEST_SET_SIZE], generator=generator))

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    validate_dataloader = DataLoader(validation_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    print('Training dataset')
    res = test(cnn, train_dataloader, device, dataset.get_encoder())
    print('Validation dataset')
    res += test(cnn, validate_dataloader, device, dataset.get_encoder())
    print('Testing dataset')
    res += test(cnn, test_dataloader, device, dataset.get_encoder())
    print('Full dataset')
    res.display_results()
