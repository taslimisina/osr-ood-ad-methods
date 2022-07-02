from typing import List

from torch.utils.data import DataLoader

class Dataset:
    def get_num_classes(self) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def get_testset(self, transform, sub_classes: List[int]):
        raise NotImplementedError

    def get_testloader(self, testset, batch_size, shuffle, num_workers, pin_memory) -> DataLoader:
        return DataLoader(testset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory)