import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable, Union, List, Tuple
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers import util


class ModularLoss(nn.Module):
    """

    """

    def __init__(self, model: SentenceTransformer,
                 loss_fct: Callable,
                 agg_fct: Callable,
                 scale: float = 20.0,
                 similarity_fct: Callable = util.cos_sim,
                 positives: int = 1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(ModularLoss, self).__init__()
        self.positives = positives
        self.similarity_fct = similarity_fct
        self.scale = scale
        self.model = model
        self.agg_fct = agg_fct
        self.loss_fct = loss_fct

    def calc_agg(self, reps):
        if self.agg_fct:
            return self.agg_fct(self, reps)
        return NotImplementedError("ModularLoss requires a valid aggregator function (agg_fct)")

    def calc_loss(self, scores, labels):
        if self.loss_fct:
            return self.loss_fct(self, scores, labels)
        return NotImplementedError("ModularLoss requires a valid loss function (loss_fct)")

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        scores, labels = self.calc_agg(reps)

        return self.calc_loss(scores * self.scale, labels)

    def split_scores_per_positive(self, scores: Tensor, labels: Tensor) -> Tensor:
        grouped_labels = labels.reshape(-1, len(scores)).transpose(0, 1)  # Examples a[i] should match with b[i + n*N]

        repeated_scores = scores.repeat(self.positives, 1)
        repeated_labels = grouped_labels.repeat(self.positives, 1)

        adjusted_labels = repeated_labels[repeated_labels != labels.unsqueeze(-1)].reshape(repeated_labels[:, 1:].shape)
        adjusted_scores = repeated_scores.scatter(1, adjusted_labels, 0)

        return adjusted_scores


#
# class BatchAggregator:
#     def __init__(self, anchors=1, positives=1):
#         self.anchors = anchors
#         self.positives = positives
#
#     def __call__(self, modular_loss: ModularLoss, reps: Union[Tuple[Tensor, ...], List[Tensor]]):


def agg_in_batch_negatives(modular_loss: ModularLoss, reps: Union[Tuple[Tensor, ...], List[Tensor]]):
    embeddings_a = reps[0]
    embeddings_b = torch.cat(reps[1:])
    scores = modular_loss.similarity_fct(embeddings_a, embeddings_b)
    labels = torch.tensor(range(len(embeddings_a) * modular_loss.positives), dtype=torch.long, device=scores.device).reshape(-1, len(scores)).transpose(0, 1)

    return scores, labels


def agg_anchors_in_batch_negatives(modular_loss: ModularLoss, reps: Union[Tuple[Tensor, ...], List[Tensor]]):
    embeddings_a = torch.cat(reps[:modular_loss.positives])
    embeddings_b = torch.cat(reps[modular_loss.positives:])
    scores = modular_loss.similarity_fct(embeddings_a, embeddings_b)
    labels = torch.tensor(range(len(embeddings_a) // modular_loss.positives), dtype=torch.long, device=scores.device).reshape(-1, len(scores)).transpose(0, 1).repeat(modular_loss.positives, 1)

    return scores, labels


def agg_all(modular_loss: ModularLoss, reps: Union[Tuple[Tensor, ...], List[Tensor]]):
    embeddings_a = reps[0]
    embeddings_b = torch.cat(reps[0:])
    scores = modular_loss.similarity_fct(embeddings_a, embeddings_b)
    labels = torch.tensor(range(len(embeddings_a) * (modular_loss.positives + 1)), dtype=torch.long, device=scores.device).reshape(-1, len(scores)).transpose(0, 1)

    return scores, labels


def agg_unique(modular_loss: ModularLoss, reps: Union[Tuple[Tensor, ...], List[Tensor]]):
    embeddings_a = reps[0]
    embeddings_b = torch.stack(reps[1:])
    scores = torch.stack([modular_loss.similarity_fct(embeddings_a[i], embeddings_b[:, i])[0] for i in range(len(embeddings_a))])
    labels = torch.tensor(range(modular_loss.positives), dtype=torch.long, device=scores.device).repeat(len(embeddings_a), 1)

    return scores, labels


def agg_positives(modular_loss: ModularLoss, reps: Union[Tuple[Tensor, ...], List[Tensor]]):
    embeddings_a = reps[0]
    embeddings_positives = torch.cat(reps[1:modular_loss.positives + 1])
    embeddings_negatives = torch.stack(reps[modular_loss.positives + 1:])
    in_batch_negatives_scores = modular_loss.similarity_fct(embeddings_a, embeddings_positives)
    uniques_scores = torch.stack([modular_loss.similarity_fct(embeddings_a[i], embeddings_negatives[:, i])[0] for i in range(len(embeddings_a))])
    scores = torch.cat((in_batch_negatives_scores, uniques_scores), dim=1)
    labels = torch.tensor(range(len(embeddings_a) * modular_loss.positives), dtype=torch.long, device=scores.device).reshape(-1, len(scores)).transpose(0, 1)

    return scores, labels


class AggregatedModularLoss(ModularLoss):
    """

    """

    def __init__(self, model: SentenceTransformer,
                 loss_fct: Callable,
                 agg_fct: Union[Callable, str] = agg_in_batch_negatives,
                 scale: float = 20.0,
                 similarity_fct: Callable = util.cos_sim,
                 positives: int = 1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(AggregatedModularLoss, self).__init__(model=model, agg_fct=agg_fct, loss_fct=loss_fct, positives=positives, scale=scale, similarity_fct=similarity_fct)
