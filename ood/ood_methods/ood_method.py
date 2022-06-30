from ood.archs.arch import Arch
from ood.scorers.scorer import Scorer


class OodMethod:
    def __init__(self, arch: Arch, scorer: Scorer):
        self.arch = arch
        self.scorer = scorer

    def get_arch(self):
        return self.arch

    def get_scorer(self):
        return self.scorer

    def get_transform(self):
        raise NotImplementedError