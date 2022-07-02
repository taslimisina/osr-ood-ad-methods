# AD/ND Benchmarks

## AUROC results for anomaly/novelty detection

The performance is averaged for each dataset in the one-vs-all setting.


| Method            | MNIST     |Fashion-MNIST  | CIFAR-10   |CIFAR-100     | MVTec AD      |
| :---------:       | :-------: |:---------:    |:---------: | :---------:  | :---------:   |
| OC-GAN            | 97.50     |-              |65.60       | -            | -             |
| LSA               | 97.50     |-              |64.10       | -            | 73.00         |
| AnoGan            | 91.30     |-              |61.79       | -            | 55.00         |
| OC-SVM            | 96.00     |92.80          |58.56       | -            | 47.90         |
| DeepSVDD          | 94.80     |92.80          |64.81       | 67.00        | 47.90         |
| GT                | -         |93.50          |86.00       | 78.70        | 67.06         |
| CSI               | -         |-              |94.30       | 89.55        | 63.60         |
| U-std             | 99.35     |-              |81.96       | -            | 85.70         |
| Multiresolution   | 98.71     |94.49          |87.18       | -            | 87.74         |
| Contrastive DA    | -         |94.50          |92.40       | 86.50        | 86.50         |