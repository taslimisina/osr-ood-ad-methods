from sklearn.metrics import average_precision_score

from osr.criteria.critic import Critic

class Aupr(Critic):
    def get_name(self):
        return 'AUPR'

    def evaluate(self, closed_set_scores, open_set_scores):
        all_scores = closed_set_scores + open_set_scores
        all_labels = [1 for _ in range(len(closed_set_scores))] + [0 for _ in range(len(open_set_scores))]
        return average_precision_score(all_labels, all_scores)