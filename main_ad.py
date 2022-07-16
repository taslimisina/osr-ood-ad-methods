from ad.evaluator import Evaluator

benchmarks = []

def main():
    for i, evaluator in enumerate(benchmarks):
        print('Benchmark', i, ':', evaluator)
        evaluator.evaluate()


if __name__ == '__main__':
    main()