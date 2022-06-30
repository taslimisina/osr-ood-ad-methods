from torchvision.transforms import transforms

from ood.archs.wrn import WideResNet
from ood.ood_methods.ood_method import OodMethod
from ood.scorers.msp_scorer import MspScorer
import torch
import requests


class Msp(OodMethod):

    def __init__(self):
        super().__init__(WideResNet(depth=40, num_classes=10), MspScorer())
        ckpt_url = 'https://github.com/hendrycks/outlier-exposure/raw/master/CIFAR/snapshots' \
                   '/baseline/cifar10_calib_wrn_baseline_epoch_99.pt'
        response = requests.get(ckpt_url)
        file = 'cifar10_wrn_baseline_epoch_99.pt'
        with open(file, 'wb') as handle:
            handle.write(response.content)
        self.arch.load_state_dict(torch.load(file))

    def get_transform(self):
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return test_transform

    def get_trained_arch(self):
        return self.arch