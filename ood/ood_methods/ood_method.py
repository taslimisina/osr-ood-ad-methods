from ood.archs.arch import Arch
from ood.datasets.dataset import Dataset
from ood.scorers.scorer import Scorer


class OodMethod:
    def __init__(self, arch: Arch, scorer: Scorer, dataset: Dataset):
        self.arch = arch
        self.scorer = scorer
        self.dataset = dataset

    def get_trained_arch(self):
        raise NotImplementedError

    def get_scorer(self):
        return self.scorer

    def get_transform(self):
        raise NotImplementedError