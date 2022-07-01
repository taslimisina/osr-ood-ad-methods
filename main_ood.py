from ood.criteria.aupr import Aupr
from ood.criteria.auroc import Auroc
from ood.criteria.fpr import Fpr
from ood.datasets.cifar100_dataset import Cifar100Dataset
from ood.datasets.cifar10_dataset import Cifar10Dataset
from ood.evaluator import Evaluator
from ood.ood_methods.msp import Msp


benchmarks = []
benchmarks.append({'ood_method': Msp(),
                   'inlier_dataset': Cifar10Dataset(),
                   'outlier_dataset': Cifar100Dataset(),
                   'criteria': [Auroc(), Aupr(), Fpr()]})

def main():
    for i, benchmark in enumerate(benchmarks):
        print('Benchmark', i, ':', benchmark)
        evaluator = Evaluator(**benchmark)
        evaluator.evaluate()


if __name__ == '__main__':
    main()