
class Dataset:

    def get_num_classes(self) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError