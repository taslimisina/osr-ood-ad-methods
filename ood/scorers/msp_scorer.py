from ood.scorers.scorer import Scorer
import torch.nn.functional as F

to_np = lambda x: x.data.cpu().numpy()

class MspScorer(Scorer):
    def get_score(self, model_output):
        score = to_np(F.softmax(model_output, dim=1))
        return score