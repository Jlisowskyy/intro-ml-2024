"""
Author: Tomasz Mycielski

Module for classifying datasets with pretrained models
"""
import torch
import torch.utils
from torch.utils.data import DataLoader

from src.cnn.loadset import MultiLabelDataset
from src.cnn.train import test
from src.constants import (TRAINING_TEST_SET_SIZE, TRAINING_VALIDATION_SET_SIZE,
                           TRAINING_TRAIN_SET_SIZE, DATABASE_ANNOTATIONS_PATH,
                           DATABASE_OUT_PATH, MODEL_BASE_PATH)
from src.model_definitions import BasicCNN
from src.model_definitions import model_definitions
from src.cnn.model_definition import ModelDefinition


def main() -> None:
    """
    Script entry point
    """

    cnn = None
    for model_definition in model_definitions:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            temp_cnn = model_definition.model()
            temp_cnn.load_model(MODEL_BASE_PATH)
            cnn = temp_cnn
            break
        except Exception as e:
            print(f'Failed to load model {model_definition.model_name}: {e}')
            continue

    if cnn is None:
        print('No model was loaded')
        return

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
