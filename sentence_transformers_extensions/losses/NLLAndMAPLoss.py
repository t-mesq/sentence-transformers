import torch
import torchsort
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from . import MultiplePositivesAndNegativesRankingLoss, map_loss, agg_in_batch_negatives


class NLLAndMAPLoss(MultiplePositivesAndNegativesRankingLoss):
    """

    """

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim, positives=1, regularization_strength=0.1, map_factor=0.5, nll_factor=0.5, agg_fct=agg_in_batch_negatives):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set sclae to 1)
        """
        super(NLLAndMAPLoss, self).__init__(model=model, scale=scale, similarity_fct=similarity_fct, positives=positives, agg_fct=agg_fct)
        self.regularization_strength = regularization_strength
        self.map_factor = map_factor
        self.nll_factor = nll_factor

    def calc_loss(self, scores, labels):
        loss_nll = super(NLLAndMAPLoss, self).calc_loss(scores, labels)
        loss_map =  map_loss(scores, labels, regularization_strength=self.regularization_strength)
        return self.nll_factor * loss_nll + self.map_factor * loss_map