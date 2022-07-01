from torchvision.datasets import CIFAR100

from ood.datasets.dataset import Dataset


class Cifar100Dataset(Dataset):
    def get_name(self) -> str:
        return 'Cifar100'

    def get_num_classes(self) -> int:
        return 100

    def get_testset(self, transform):
        return CIFAR100('./cifar100', train=False, transform=transform, download=True)