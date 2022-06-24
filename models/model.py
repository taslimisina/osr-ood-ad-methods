from datasets.dataset import Dataset
from typing import List

class Model:

    def train_ad(self, dataset: Dataset, normal_class: int) -> None:
        raise NotImplementedError

    def test_ad(self, dataset: Dataset, normal_class: int):
        raise NotImplementedError


    def train_osr(self, dataset: Dataset, normal_set: List[int]) -> None:
        raise NotImplementedError

    def test_osr(self, dataset: Dataset, normal_set: List[int]):
        raise NotImplementedError


    def train_ood(self, dataset: Dataset) -> None:
        raise NotImplementedError

    def test_ood(self, dataset: Dataset, outliers: List[Dataset]):
        raise NotImplementedError