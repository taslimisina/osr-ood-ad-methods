# OSR Benchmarks

## AUROC results for Open-Set Recognition

The performance is averaged across each dataset for 10 random trials.

| Method            | MNIST     |SVHN           | CIFAR-10   |CIFAR+10      | CIFAR+50      |TinyImageNet   |
| :---------:       | :-------: |:---------:    |:---------: | :---------:  | :---------:   |:---------:    |
| OpenMax           | 98.10     |89.40          |81.10       | 81.70        | 79.60         |81.10          |
| G-OpenMax         | 98.40     |89.60          |67.60       | 82.70        | 81.90         |58.00          |
| OSRCI             | 98.80     |91.00          |69.90       | 83.80        | 82.70         |58.60          |
| C2AE              | 98.90     |92.20          |89.50       | 95.50        | 93.70         |74.80          |
| CROSR             | 99.20     |89.90          |88.30       | -            | -             |58.90          |
| GDFR              | -         |93.50          |80.70       | 92.80        | 92.60         |60.80          |
| RPL               | 99.60     |96.80          |90.10       | 97.60        | 96.80         |80.90          |
| OpenGan           | 99.90     |98.80          |97.30       | -            | -             |90.70          |
| MLS               | 99.30     |97.10          |93.60       | 97.90        | 96.50         |83.00          |