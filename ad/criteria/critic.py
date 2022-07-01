

class Critic:

    def evaluate(self, normal_scores, anomaly_scores):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError