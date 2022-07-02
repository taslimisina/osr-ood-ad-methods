from sklearn.metrics import average_precision_score

from ad.criteria.critic import Critic

class Aupr(Critic):
    def get_name(self):
        return 'AUPR'

    def evaluate(self, normal_scores, anomaly_scores):
        all_scores = normal_scores + anomaly_scores
        all_labels = [1 for _ in range(len(normal_scores))] + [0 for _ in range(len(anomaly_scores))]
        return average_precision_score(all_labels, all_scores)