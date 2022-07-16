from torchvision.transforms import transforms

from ood.archs.wrn import WideResNet
from ood.ood_methods.ood_method import OodMethod
from ood.scorers.msp_scorer import MspScorer
import torch
import requests


ckpt_urls = {
    'Cifar10': 'https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots'
               '/baseline/cifar10_calib_wrn_baseline_epoch_99.pt',
    'Cifar100': 'https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots'
                '/baseline/cifar100_calib_wrn_baseline_epoch_99.pt'
}

class Msp(OodMethod):

    def __init__(self):
        super().__init__(WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3), MspScorer())

    def get_transform(self):
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return test_transform

    def get_trained_arch(self, dataset):
        if dataset not in ckpt_urls.keys():
            print(dataset, 'dataset is not available!')
            return
        ckpt_url = ckpt_urls[dataset]
        response = requests.get(ckpt_url)
        file = 'msp_checkpoint.pt'
        with open(file, 'wb') as handle:
            handle.write(response.content)
        self.arch.load_state_dict(torch.load(file))
        return self.arch

    def __str__(self):
        return 'MSP'