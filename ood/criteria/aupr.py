from sklearn.metrics import average_precision_score

from ood.criteria.critic import Critic

class Aupr(Critic):
    def get_name(self):
        return 'AUPR'

    def evaluate(self, inlier_scores, outlier_scores):
        all_scores = inlier_scores + outlier_scores
        all_labels = [1 for _ in range(len(inlier_scores))] + [0 for _ in range(len(outlier_scores))]
        return average_precision_score(all_labels, all_scores)