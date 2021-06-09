from sentence_transformers import util, SentenceTransformer
from sentence_transformers.losses import *
import unittest
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from ml_metrics import apk

EPSILON = 0.001


class RankingLossestest(unittest.TestCase):
    def setUp(self):
        self.scores = torch.rand(1000, 1000)
        self.relevance = torch.rand(1000, 1000)
        self.labels = torch.argsort(self.relevance, descending=True)[:, :100]

    def test_mean_average_precision_loss(self):
        """Tests the correct computation of MAP score"""

        loss = MeanAveragePrecisionLoss(None, regularization_strength=0.0000001)

        for i, (scores_, relevance_) in enumerate(zip(self.scores, self.labels)):
            loss_result = (float(loss.map(scores_.unsqueeze(0), relevance_.unsqueeze(0))))
            real_result = 1 - apk(list(relevance_.numpy()), list(torch.argsort(scores_, descending=True).numpy()), k=1000)
            assert abs(loss_result - real_result) < EPSILON

    def test_nrmalized_discounted_cumulative_gain_loss(self):
        """Tests the correct computation of NDCG score"""

        loss = NormalizedDiscountedCumulativeGainLoss(None, regularization_strength=0.0000001)

        for i, (scores_, relevance_) in enumerate(zip(self.scores, self.relevance)):
            loss_result = (float(loss.ndcg_loss(scores_.unsqueeze(0), relevance_.unsqueeze(0), reduction="mean")))
            real_result = (1 - ndcg_score([relevance_.numpy()], [scores_.numpy()]))
            assert abs(loss_result - real_result) < EPSILON

if "__main__" == __name__:
    unittest.main()