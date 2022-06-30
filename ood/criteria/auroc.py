from ood.criteria.critic import Critic
from sklearn.metrics import roc_auc_score

class Auroc(Critic):
    def get_name(self):
        return 'AUROC'

    def evaluate(self, inlier_scores, outlier_scores):
        all_scores = inlier_scores + outlier_scores
        all_labels = [1 for _ in range(len(inlier_scores))] + [0 for _ in range(len(outlier_scores))]
        return roc_auc_score(all_labels, all_scores)