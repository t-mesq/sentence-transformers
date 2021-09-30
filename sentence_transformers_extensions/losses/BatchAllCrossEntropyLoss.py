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

    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True):
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

    def get_all_possible_scores(self, labels_a: Tensor, embeddings_a: Tensor, labels_b: Tensor = None, embeddings_b: Tensor = None):
        if labels_b is None:
            labels_b = labels_a
            embeddings_b = embeddings_a
        # Compute the similarity function over all pairs
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # Computes a boolean adjacency matrix for all valid pairs (labels match)
        label_equal = labels_a.unsqueeze(1) == labels_b.unsqueeze(0)
        if torch.equal(embeddings_a, embeddings_b) and (not self.diagonal):  # Remove diagonal?
            label_equal[torch.eye(labels_a.size(0), device=labels_a.device).bool()] = False
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


class RankingBatchAllCrossEntropyLoss(BatchAllCrossEntropyLoss):
    """
    Same as BatchAllCrossEntropyLoss but adapted for ranking examples
    """

    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True):
        """

        :param diagonal: Use pairs in the diagonal
        :param model: SentenceTransformer model
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :param scale: Output of similarity function is multiplied by scale value
        :param loss_fct: Loss function to be applied, must take a 2d tensor for the scores, and a 1d tensor for the labels, corresponding to the index of the positive
        """
        super(RankingBatchAllCrossEntropyLoss, self).__init__(model=model, similarity_fct=similarity_fct, scale=scale, loss_fct=loss_fct, diagonal=diagonal)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings = torch.cat(reps)
        flat_labels = torch.flatten(labels)

        scores, labels = self.get_all_possible_scores(flat_labels, embeddings)

        return self.loss_fct(scores, labels)


class RankingBatchSplitCrossEntropyLoss(RankingBatchAllCrossEntropyLoss):
    """
    Same as BatchAllCrossEntropyLoss but adapted for ranking examples
    """

    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True):
        """

        :param diagonal: Use pairs in the diagonal
        :param model: SentenceTransformer model
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :param scale: Output of similarity function is multiplied by scale value
        :param loss_fct: Loss function to be applied, must take a 2d tensor for the scores, and a 1d tensor for the labels, corresponding to the index of the positive
        """
        super(RankingBatchSplitCrossEntropyLoss, self).__init__(model=model, similarity_fct=similarity_fct, scale=scale, loss_fct=loss_fct, diagonal=diagonal)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        query_reps = []
        query_labels = []
        document_reps = []
        document_labels = []

        for sentence_feature, label in zip(sentence_features, labels):
            rep = self.model(sentence_feature)
            if rep['encoder'] == 'query':
                query_reps.append(rep['sentence_embedding'])
                query_labels.append(label)

            else:
                document_reps.append(rep['sentence_embedding'])
                document_labels.append(label)

        query_embeddings = torch.cat(query_reps)
        document_embeddings = torch.cat(document_reps)

        query_labels = torch.cat(query_labels)
        document_labels = torch.cat(document_labels)

        return self.get_embeddings_combination(query_labels, query_embeddings, document_labels, document_embeddings)

    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        raise NotImplementedError


class RankingBatchQueriesCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        scores, labels = self.get_all_possible_scores(query_labels, query_embeddings, torch.cat((query_labels, document_labels)), torch.cat((query_embeddings, document_embeddings)))
        return self.loss_fct(scores, labels)


class RankingBatchDocumentsCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        scores, labels = self.get_all_possible_scores(torch.cat((query_labels, document_labels)), torch.cat((query_embeddings, document_embeddings)), document_labels, document_embeddings)
        return self.loss_fct(scores, labels)


class RankingBatchSingleCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        scores, labels = self.get_all_possible_scores(query_labels, query_embeddings, document_labels, document_embeddings)
        return self.loss_fct(scores, labels)


class RankingBatchTripleCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True, alpha: float = 0.5):
        super(RankingBatchTripleCrossEntropyLoss, self).__init__(model=model, similarity_fct=similarity_fct, scale=scale, loss_fct=loss_fct, diagonal=diagonal)
        self.alpha = alpha

    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        ranking_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings, document_labels, document_embeddings))
        queries_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings))
        documents_loss = self.loss_fct(*self.get_all_possible_scores(document_labels, document_embeddings))
        return self.alpha * ranking_loss + (1 - self.alpha) * (0.5 * queries_loss + 0.5 * documents_loss)


class RankingBatchQuadrupleCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True, alpha: float = 0.5):
        super(RankingBatchQuadrupleCrossEntropyLoss, self).__init__(model=model, similarity_fct=similarity_fct, scale=scale, loss_fct=loss_fct, diagonal=diagonal)
        self.alpha = alpha

    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        inverted_ranking_loss = self.loss_fct(*self.get_all_possible_scores(document_labels, document_embeddings, query_labels, query_embeddings))
        ranking_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings, document_labels, document_embeddings))
        queries_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings))
        documents_loss = self.loss_fct(*self.get_all_possible_scores(document_labels, document_embeddings))
        return self.alpha * (0.5 * ranking_loss + 0.5 * inverted_ranking_loss) + (1 - self.alpha) * (0.5 * queries_loss + 0.5 * documents_loss)


class RankingBatchDoubleQueryCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True, alpha: float = 0.5):
        super(RankingBatchDoubleQueryCrossEntropyLoss, self).__init__(model=model, similarity_fct=similarity_fct, scale=scale, loss_fct=loss_fct, diagonal=diagonal)
        self.alpha = alpha

    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        ranking_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings, document_labels, document_embeddings))
        queries_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings))
        return self.alpha * ranking_loss + (1 - self.alpha) * queries_loss


class RankingBatchDoubleDocumentCrossEntropyLoss(RankingBatchSplitCrossEntropyLoss):
    def __init__(self, model: SentenceTransformer, similarity_fct: Callable = util.cos_sim, scale: float = 20.0, loss_fct: Callable = nn.CrossEntropyLoss(), diagonal: bool = True, alpha: float = 0.5):
        super(RankingBatchDoubleDocumentCrossEntropyLoss, self).__init__(model=model, similarity_fct=similarity_fct, scale=scale, loss_fct=loss_fct, diagonal=diagonal)
        self.alpha = alpha

    def get_embeddings_combination(self, query_labels, query_embeddings, document_labels, document_embeddings):
        ranking_loss = self.loss_fct(*self.get_all_possible_scores(query_labels, query_embeddings, document_labels, document_embeddings))
        documents_loss = self.loss_fct(*self.get_all_possible_scores(document_labels, document_embeddings))
        return self.alpha * ranking_loss + (1 - self.alpha) * documents_loss
