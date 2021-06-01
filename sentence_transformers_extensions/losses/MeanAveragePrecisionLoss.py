import torch
import torchsort
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from . import AggregatedModularLoss, agg_in_batch_negatives


def map_loss(pred, labels, regularization_strength=0.1):
    sorted_preds = torchsort.soft_rank(-1 * pred, regularization_strength=regularization_strength)
    denominator = torch.gather(sorted_preds, 1, labels)
    numerator = torch.argsort(torch.argsort(denominator)) + 1  # Examples a[i] should match with b[i + n*N]
    ap = numerator / denominator
    map = torch.mean(ap)
    return 1 - map


class MeanAveragePrecisionLoss(AggregatedModularLoss):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
            from sentence_transformers.readers import InputExample

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim, positives: int = 1, regularization_strength=0.1, agg_fct=agg_in_batch_negatives):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set sclae to 1)
        """
        super(MeanAveragePrecisionLoss, self).__init__(model=model, scale=scale, similarity_fct=similarity_fct, agg_fct=agg_fct, loss_fct=None, positives=positives)
        self.regularization_strength = regularization_strength

    def calc_loss(self, scores, labels):
        return map_loss(scores, labels, regularization_strength=self.regularization_strength)
