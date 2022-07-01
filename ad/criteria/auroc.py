from ad.criteria.critic import Critic
from sklearn.metrics import roc_auc_score

class Auroc(Critic):
    def get_name(self):
        return 'AUROC'

    def evaluate(self, normal_scores, anomaly_scores):
        all_scores = normal_scores + anomaly_scores
        all_labels = [1 for _ in range(len(normal_scores))] + [0 for _ in range(len(anomaly_scores))]
        return roc_auc_score(all_labels, all_scores)