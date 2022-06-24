# import typing
from models.model import Model
from datasets.dataset import Dataset
from typing import List


def evaluate_ad(model: Model, dataset: Dataset):
    for normal_class in range(dataset.get_num_classes()):
        model.train_ad(dataset, normal_class)
        result = model.test_ad(dataset, normal_class)


def evaluate_ood(model: Model, dataset: Dataset, outliers: List[Dataset]):
    model.train_ood(dataset)
    result = model.test_ood(dataset, outliers)