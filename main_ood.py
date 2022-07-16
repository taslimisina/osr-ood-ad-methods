from ood.criteria.aupr import Aupr
from ood.criteria.auroc import Auroc
from ood.criteria.fpr import Fpr
from ood.datasets.cifar100_dataset import Cifar100Dataset
from ood.datasets.cifar10_dataset import Cifar10Dataset
from ood.evaluator import Evaluator
from ood.ood_methods.msp import Msp
from ood.ood_methods.msp_oe import MspOE


cifar10 = Cifar10Dataset()
cifar100 = Cifar100Dataset()
criteria = [Auroc(), Aupr(), Fpr()]

benchmarks = []
# Cifar10
benchmarks.extend([
    Evaluator(Msp(cifar10), cifar10, cifar100, criteria),
    Evaluator(MspOE(cifar10), cifar10, cifar100, criteria)
])
# Cifar100
benchmarks.extend([
    Evaluator(Msp(cifar100), cifar100, cifar10, criteria),
    Evaluator(MspOE(cifar100), cifar100, cifar10, criteria)
])

def main():
    for i, evaluator in enumerate(benchmarks):
        print('Benchmark', i, ':', evaluator)
        evaluator.evaluate()


if __name__ == '__main__':
    main()