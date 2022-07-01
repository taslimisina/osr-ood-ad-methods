from ad.evaluator import Evaluator

benchmarks = []

def main():
    for i, benchmark in enumerate(benchmarks):
        print('Benchmark', i, ':', benchmark)
        evaluator = Evaluator(**benchmark)
        evaluator.evaluate()


if __name__ == '__main__':
    main()