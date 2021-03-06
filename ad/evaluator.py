from typing import List

import torch

from ad.ad_methods.ad_method import AdMethod
from ad.criteria.critic import Critic
from ad.datasets.dataset import Dataset


class Evaluator:

    def __init__(self, ad_method: AdMethod, dataset: Dataset, criteria: List[Critic],
                 batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        self.ad_method = ad_method
        self.dataset = dataset
        self.criteria = criteria
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @torch.no_grad()
    def evaluate(self):
        model = self.ad_method.get_trained_arch()
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        num_classes = self.dataset.get_num_classes()
        normal_class = self.ad_method.get_normal_class()
        anomaly_classes = [i for i in range(num_classes) if i != normal_class]

        normal_testset = self.dataset.get_testset(self.ad_method.get_transform(), [normal_class])
        normal_testloader = self.dataset.get_testloader(
            normal_testset, self.batch_size, self.shuffle, self.num_workers, self.pin_memory)

        anomaly_testset = self.dataset.get_testset(self.ad_method.get_transform(), anomaly_classes)
        anomaly_testloader = self.dataset.get_testloader(
            anomaly_testset, self.batch_size, self.shuffle, self.num_workers, self.pin_memory)

        normal_scores = []
        for data, target in normal_testloader:
            data.to(device)
            normal_output = model(data)
            scores = self.ad_method.get_scorer().get_score(normal_output)
            normal_scores.extend(scores)

        anomaly_scores = []
        for data, target in anomaly_testloader:
            data.to(device)
            anomaly_output = model(data)
            scores = self.ad_method.get_scorer().get_score(anomaly_output)
            anomaly_scores.extend(scores)

        print('Normal Class:', normal_class)
        for i, critic in enumerate(self.criteria):
            eval_score = critic.evaluate(normal_scores, anomaly_scores)
            print(critic.get_name(), eval_score)

    def __str__(self):
        return 'ad_method: ' + str(self.ad_method) + '  ' + \
            'dataset: ' + self.dataset.get_name() + '  ' + \
            'normal_class: ' + str(self.ad_method.get_normal_class()) + '  ' + \
            'criteria: ' + str([critic.get_name() for critic in self.criteria]) + '  ' + \
            'batch_size: ' + str(self.batch_size)