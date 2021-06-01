from typing import Iterable, Dict

import torch
import torchsort
from torch import Tensor
from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from . import AggregatedModularLoss, agg_in_batch_negatives


def ndcg_loss(scores, relevance, regularization_strength=0.1, k=None, reduction="mean", **kw):
    rel_target = torch.argsort(torch.argsort(-1.0 * relevance))
    rel_pred = torchsort.soft_rank(-1.0 * scores, regularization_strength=regularization_strength) - 1.0

    p = len(rel_target[0]) if k is None else k

    rel_target_mask = rel_target < p
    rel_pred_mask = torch.argsort(torch.argsort(-1.0 * scores)) < p

    idcg = torch.sum((relevance[rel_target_mask] / torch.log2(rel_target[rel_target_mask] + 2.0)).reshape(-1, p), dim=1)
    dcg = torch.sum((relevance[rel_pred_mask] / torch.log2(rel_pred[rel_pred_mask] + 2.0)).reshape(-1, p), dim=1)

    if reduction == 'mean':
        return torch.mean(1.0 - (dcg / idcg))
    elif reduction == 'sum':
        return torch.sum(1.0 - (dcg / idcg))
    else:
        return 1.0 - (dcg / idcg)

class NormalizedDiscountedCumulativeGainLoss(AggregatedModularLoss):
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
        super(NormalizedDiscountedCumulativeGainLoss, self).__init__(model=model, scale=scale, similarity_fct=similarity_fct, agg_fct=agg_fct, loss_fct=None, positives=positives)
        self.regularization_strength = regularization_strength

    def calc_loss(self, scores, labels):
        relevance = torch.zeros_like(scores).scatter(1, labels, 1)
        return ndcg_loss(scores, relevance, regularization_strength=self.regularization_strength)



# import torch
# import torchsort
# scores = torch.tensor( [[ 6., 5., 4., 3., 2., 1. ] , [ 6., 5., 4., 3., 2., 1. ] ] , requires_grad=True)
# relevance = torch.tensor( [[ 3, 2, 3, 0, 1, 2 ] , [ 3, 3, 3, 0, 1, 2 ] ] )
# def ndcg_loss(scores, relevance, k=None, reduction="mean", **kw):
#   rel_target = -1.0 * torchsort.soft_sort(-1.0 * relevance)
#   pred = torchsort.soft_rank(-1.0 * scores, **kw) - 1.0
#   rel_pred = scores
#   for p1, i1 in enumerate(pred):
#     for p2, i2 in enumerate(i1):
#       rel_pred[p1,p2] = relevance[p1,i2.long()]
#   if k is None: p = min(len(rel_target[0]), len(rel_pred[0]))
#   else: p = min(len(rel_target[0]), min(len(rel_pred[0]), k))
#   discount = 1.0 / (torch.log2(torch.arange(p) + 2.0))
#   idcg = torch.sum(rel_target[:p] * discount, dim=1)
#   dcg = torch.sum(rel_pred[:p] * discount, dim=1)
#   if reduction == 'mean': return torch.mean(1.0 - (dcg / idcg))
#   elif reduction == 'sum': return torch.sum(1.0 - (dcg / idcg))
#   else: return 1.0 - (dcg / idcg)
# result = ndcg_loss( scores, relevance, reduction="mean" )
# print(result)
# grad = torch.autograd.grad(result , scores)
# print(grad )