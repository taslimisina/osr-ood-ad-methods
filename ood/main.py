from ood.criteria.auroc import Auroc
from ood.datasets.cifar100_dataset import Cifar100Dataset
from ood.datasets.cifar10_dataset import Cifar10Dataset
from ood.evaluator import Evaluator
from ood.ood_methods.msp import Msp


def main():
    evaluator = Evaluator(Msp(), Cifar10Dataset(), Cifar100Dataset(), [Auroc()])
    evaluator.evaluate()


if __name__ == '__main__':
    main()