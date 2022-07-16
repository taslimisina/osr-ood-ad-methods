from ood.criteria.aupr import Aupr
from ood.criteria.auroc import Auroc
from ood.criteria.fpr import Fpr
from ood.datasets.cifar100_dataset import Cifar100Dataset
from ood.datasets.cifar10_dataset import Cifar10Dataset
from ood.evaluator import Evaluator
from ood.ood_methods.msp import Msp
from ood.ood_methods.msp_oe import MspOE

benchmarks = []

# Cifar10
benchmarks.append(Evaluator(Msp(), Cifar10Dataset(), Cifar100Dataset(), [Auroc(), Aupr(), Fpr()]))
benchmarks.append(Evaluator(MspOE(), Cifar10Dataset(), Cifar100Dataset(), [Auroc(), Aupr(), Fpr()]))

# Cifar100
benchmarks.append(Evaluator(Msp(), Cifar100Dataset(), Cifar10Dataset(), [Auroc(), Aupr(), Fpr()]))
benchmarks.append(Evaluator(MspOE(), Cifar100Dataset(), Cifar10Dataset(), [Auroc(), Aupr(), Fpr()]))

def main():
    for i, evaluator in enumerate(benchmarks):
        print('Benchmark', i, ':', evaluator)
        evaluator.evaluate()


if __name__ == '__main__':
    main()