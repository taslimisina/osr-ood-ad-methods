from osr.criteria.critic import Critic
import numpy as np

class Fpr(Critic):

    def __init__(self, recall_level=0.95):
        super().__init__()
        self.recall_level = recall_level

    def get_name(self):
        return 'FPR(' + str(self.recall_level * 100) + ')'

    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                               'its last element does not correspond to sum')
        return out

    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                         np.array_equal(classes, [-1, 1]) or
                         np.array_equal(classes, [0]) or
                         np.array_equal(classes, [-1]) or
                         np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = self.stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

        thresholds = y_score[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1]
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

        cutoff = np.argmin(np.abs(recall - recall_level))

        return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

    def evaluate(self, closed_set_scores, open_set_scores):
        all_scores = closed_set_scores + open_set_scores
        all_labels = [1 for _ in range(len(closed_set_scores))] + [0 for _ in range(len(open_set_scores))]
        return self.fpr_and_fdr_at_recall(np.array(all_labels), np.array(all_scores), self.recall_level)