from ad.archs.arch import Arch
from ad.scorers.scorer import Scorer


class AdMethod:
    def __init__(self, arch: Arch, scorer: Scorer):
        self.arch = arch
        self.scorer = scorer

    def get_trained_arch(self, dataset: str):
        raise NotImplementedError

    def get_normal_class(self) -> int:
        raise NotImplementedError

    def get_scorer(self):
        return self.scorer

    def get_transform(self):
        raise NotImplementedError