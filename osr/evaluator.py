from typing import List

import torch

from osr.criteria.critic import Critic
from osr.datasets.dataset import Dataset
from osr.osr_methods.osr_method import OsrMethod


class Evaluator:

    def __init__(self, osr_method: OsrMethod, dataset: Dataset, criteria: List[Critic],
                 batch_size=32, shuffle=False, num_workers=2, pin_memory=True):
        self.osr_method = osr_method
        self.dataset = dataset
        self.criteria = criteria
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @torch.no_grad()
    def evaluate(self):
        model = self.osr_method.get_trained_arch(self.dataset.get_name())
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        num_classes = self.dataset.get_num_classes()
        closed_set = self.osr_method.get_closed_set()
        open_set = [i for i in range(num_classes) if i not in closed_set]

        closed_testset = self.dataset.get_testset(self.osr_method.get_transform(), closed_set)
        closed_testloader = self.dataset.get_testloader(
            closed_testset, self.batch_size, self.shuffle, self.num_workers, self.pin_memory)

        open_testset = self.dataset.get_testset(self.osr_method.get_transform(), open_set)
        open_testloader = self.dataset.get_testloader(
            open_testset, self.batch_size, self.shuffle, self.num_workers, self.pin_memory)

        closed_scores = []
        for data, target in closed_testloader:
            data.to(device)
            closed_output = model(data)
            scores = self.osr_method.get_scorer().get_score(closed_output)
            closed_scores.extend(scores)

        open_scores = []
        for data, target in open_testloader:
            data.to(device)
            open_output = model(data)
            scores = self.osr_method.get_scorer().get_score(open_output)
            open_scores.extend(scores)

        print('Closed Set:', closed_set)
        for i, critic in enumerate(self.criteria):
            eval_score = critic.evaluate(closed_scores, open_scores)
            print(critic.get_name(), eval_score)

    def __str__(self):
        return 'osr_method: ' + str(self.osr_method) + '\t' + \
            'dataset: ' + self.dataset.get_name() + '\t' + \
            'closed_set: ' + str(self.osr_method.get_closed_set()) + '\t' + \
            'criteria: ' + str(self.criteria) + '\t' + \
            'batch_size: ' + str(self.batch_size)