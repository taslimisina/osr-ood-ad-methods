from typing import List

from osr.archs.arch import Arch
from osr.datasets.dataset import Dataset
from osr.scorers.scorer import Scorer


class OsrMethod:
    def __init__(self, arch: Arch, scorer: Scorer, dataset: Dataset):
        self.arch = arch
        self.scorer = scorer
        self.dataset = dataset

    def get_trained_arch(self):
        raise NotImplementedError

    def get_closed_set(self) -> List[int]:
        raise NotImplementedError

    def get_scorer(self):
        return self.scorer

    def get_transform(self):
        raise NotImplementedError