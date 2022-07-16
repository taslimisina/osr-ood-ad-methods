from osr.evaluator import Evaluator
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

benchmarks = []

def main():
    for i, evaluator in enumerate(benchmarks):
        print('Benchmark', i, ':', evaluator)
        evaluator.evaluate()


if __name__ == '__main__':
    main()