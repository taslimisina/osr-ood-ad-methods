from torchvision.datasets import CIFAR10

from ood.datasets.dataset import Dataset


class Cifar10Dataset(Dataset):
    def get_name(self) -> str:
        return 'Cifar10'

    def get_num_classes(self) -> int:
        return 10

    def get_testset(self, transform):
        return CIFAR10('./data/cifar10', train=False, transform=transform, download=True)