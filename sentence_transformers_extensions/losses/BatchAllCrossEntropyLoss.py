import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers import util
from sentence_transformers_extensions.losses import ModularLoss


class BatchAllCrossEntropyLoss(nn.Module):
    """
    BatchAllCrossEntropyLoss takes a batch with (label, sentence) pairs and computes the cross-entropy loss for all possible, valid
    pairs, with negatives, i.e., anchor and positive must have the same label, anchor and negatives a different label. The labels
    must be integers, with same label indicating sentences from the same class. Your train dataset
    must contain at least 2 examples per label class.

    """

    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = False):
        """

        :param diagonal: Use pairs in the diagonal
        :param model: SentenceTransformer model
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :param scale: Output of similarity function is multiplied by scale value
        :param loss_fct: Loss function to be applied, must take a 2d tensor for the scores, and a 1d tensor for the labels, corresponding to the index of the positive
        """
        super(BatchAllCrossEntropyLoss, self).__init__()
        self.diagonal = diagonal
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = self.model(sentence_features[0])['sentence_embedding']
        scores, labels = self.get_all_possible_scores(labels, embeddings)

        return self.loss_fct(scores, labels)

    def get_all_possible_scores(self, labels: Tensor, embeddings: Tensor):
        # Compute the similarity function over all pairs
        scores = self.similarity_fct(embeddings, embeddings) * self.scale
        # Computes a boolean adjacency matrix for all valid pairs (labels match)
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        if not self.diagonal:  # Remove diagonal?
            label_equal[torch.eye(labels.size(0), device=labels.device).bool()] = False
        label_ids = label_equal.nonzero()
        # computes the number of possible positives, for each embedding
        line_ids = label_ids[:, 0]
        # repeats each line of scores per positive
        repeated_scores = torch.index_select(scores, index=line_ids, dim=0)
        # the correct label of each line
        adjusted_labels = label_ids[:, 1]

        # create a mask for the omitted positives in each line
        repeated_label_equal = torch.index_select(label_equal, index=line_ids, dim=0)
        labels_mask = repeated_label_equal.scatter(1, adjusted_labels.unsqueeze(1), False)
        adjusted_scores = repeated_scores
        adjusted_scores[labels_mask] = float('-inf')

        return adjusted_scores, adjusted_labels
