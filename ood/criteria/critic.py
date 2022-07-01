

class Critic:

    def evaluate(self, inlier_scores, outlier_scores):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError