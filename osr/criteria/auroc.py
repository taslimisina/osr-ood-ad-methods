from osr.criteria.critic import Critic
from sklearn.metrics import roc_auc_score

class Auroc(Critic):
    def get_name(self):
        return 'AUROC'

    def evaluate(self, closed_set_scores, open_set_scores):
        all_scores = closed_set_scores + open_set_scores
        all_labels = [1 for _ in range(len(closed_set_scores))] + [0 for _ in range(len(open_set_scores))]
        return roc_auc_score(all_labels, all_scores)