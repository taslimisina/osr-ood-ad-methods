# OSR OOD AD Methods

This is the official repository for the paper [A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges](https://arxiv.org/abs/2110.14051).

![OSR_OOD_AD](OSR_OOD_AD.jpg)

In this repo, we aim to benchmark different methods proposed for Anomaly Detection, Novelty Detection, Out-of-Distribution Detection, and Open-Set Recognition.

A curated list of these methods are available at [here](https://github.com/hoya012/awesome-anomaly-detection) and [here](https://github.com/iCGY96/awesome_OpenSetRecognition_list#open-set-recognition).


## Running the code
To benchmark *OOD*, *OSR* or *AD/ND* methods, run *main_ood.py*, *main_osr.py* or *main_ad.py* respectively.
You can modify these files to comment the benchmarks which you don't wish to run or to create a custom benchmark.

The repo is under construction and more methods and benchmarks will be added soon. We sincerely welcome you to add your methods and pre-trained models and create pull requests.
For contribution please refer to [Code Structure and Contribution](#code-structure-and-contribution).


## Datasets

### 1. Semantic-level Datasets
- MNIST
- Fashion MNIST
- CIFAR-10
- CIFAR-100
- TinyImageNet
- LSUN
- COIL-100
- SVHN
- Flowers
- Birds


### 2. Pixel-level Datasets
- MVTec AD
- PCB
- LaceAD
- Retinal-OCT
- CAMELYON16
- Chest X-Rays
- Species
- ImageNet-O


### 3. Synthetic Datasets
- MNIST-C
- ImageNet-C, ImageNet-P


## Evaluation
- AUC-ROC
- FPR@TPR
- AUPR
- Accuracy
- F-measure


## Code Structure and Contribution
For each taxonomy of OOD, OSR and AD/ND there are a main file and a folder correspondingly.
Below is the code structure for OOD. The structure for OSR and AD/ND is similar.

```
main_ood.py
ood
|   evaluator.py
|
└───archs
|   |   arch.py
|
└───criteria
|   |   critic.py
|
└───datasets
|   |   dataset.py
|
└───ood_methods
|   |   ood_method.py
|
└───scorers
    |   scorer.py
```

archs: model architectures, e.g. WideResNet model

criteria: evaluation criteria, e.g. AUROC

datasets: datasets used for inlier and outlier data, e.g. CIFAR10/100

ood_methods: OOD methods, e.g. MSP

scorers: functions that score the model's output for an ood method, e.g. MSP scorer

The python files in subfolders shown in the code structure above are interfaces for which you can add a class that inherits them and overrides their methods.
For example, if you want to add an outlier dataset to run the benchmarks with, you can add a class in the *datasets* folder and inherit the *Dataset* class in *dataset.py*.

As this repo is for evaluation, for adding a method, please do not provide code for training and also do not commit model checkpoints as it makes the repo too big. Instead, please upload your pre-trained model on the internet and download and load the model in the code.