

class Critic:

    def evaluate(self, closed_set_scores, open_set_scores):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError