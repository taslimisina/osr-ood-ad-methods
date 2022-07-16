from typing import List

import torch

from ood.criteria.critic import Critic
from ood.datasets.dataset import Dataset
from ood.ood_methods.ood_method import OodMethod


class Evaluator:

    def __init__(self, ood_method: OodMethod, inlier_dataset: Dataset, outlier_dataset: Dataset,
                 criteria: List[Critic], batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        self.ood_method = ood_method
        self.inlier_dataset = inlier_dataset
        self.outlier_dataset = outlier_dataset
        self.criteria = criteria
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @torch.no_grad()
    def evaluate(self):
        model = self.ood_method.get_trained_arch()
        inlier_testset = self.inlier_dataset.get_testset(self.ood_method.get_transform())
        inlier_testloader = self.inlier_dataset.get_testloader(
            inlier_testset, self.batch_size, self.shuffle, self.num_workers, self.pin_memory)
        outlier_testset = self.outlier_dataset.get_testset(self.ood_method.get_transform())
        outlier_testloader = self.outlier_dataset.get_testloader(
            outlier_testset, self.batch_size, self.shuffle, self.num_workers, self.pin_memory)

        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inlier_scores = []
        for data, target in inlier_testloader:
            data.to(device)
            inlier_output = model(data)
            scores = self.ood_method.get_scorer().get_score(inlier_output)
            inlier_scores.extend(scores)

        outlier_scores = []
        for data, target in outlier_testloader:
            data.to(device)
            outlier_output = model(data)
            scores = self.ood_method.get_scorer().get_score(outlier_output)
            outlier_scores.extend(scores)

        for critic in self.criteria:
            print(critic.get_name(), critic.evaluate(inlier_scores, outlier_scores))

    def __str__(self):
        return 'ood_method: ' + str(self.ood_method) + '  ' + \
            'inlier_dataset: ' + self.inlier_dataset.get_name() + '  ' + \
            'outlier_dataset: ' + self.outlier_dataset.get_name() + '  ' + \
            'criteria: ' + str([critic.get_name() for critic in self.criteria]) + '  ' + \
            'batch_size: ' + str(self.batch_size)