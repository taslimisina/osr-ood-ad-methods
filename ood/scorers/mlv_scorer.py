from ood.scorers.scorer import Scorer
import torch.nn.functional as F
import numpy as np

to_np = lambda x: x.data.cpu().numpy()

class MlvScorer(Scorer):
    def get_score(self, model_output):
        score = to_np(model_output)
        score = np.max(score, axis=1)
        return score