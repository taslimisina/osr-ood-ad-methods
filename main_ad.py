from ad.evaluator import Evaluator
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

benchmarks = []

def main():
    for i, evaluator in enumerate(benchmarks):
        print('\nBenchmark', i, ':')
        print(evaluator)
        evaluator.evaluate()


if __name__ == '__main__':
    main()