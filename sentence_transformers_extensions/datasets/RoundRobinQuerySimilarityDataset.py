"""

"""
import math
from collections import defaultdict

import numpy as np

from sentence_transformers import InputExample
from . import RoundRobinRankingDataset


class RoundRobinQuerySimilarityDataset(RoundRobinRankingDataset):
    def __init__(self, model, queries, rel_queries, rel_corpus, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, neg_rel_corpus=None, replace=True):
        super().__init__(model, queries, None, rel_queries, rel_corpus, None, batch_size, n_positives, temperature, shuffle, n_negatives, neg_rel_corpus)
        self.replace = replace
        self.sample_sizes = [self.n_positives] * (self.batch_size // self.n_positives)
        self.sample_sizes[-1] += self.batch_size % self.n_positives
        # frequency_mask = self.rel_queries.str.len() >= self.n_positives
        # self.rel_queries = self.rel_queries[frequency_mask]
        # self.weights = self.weights[frequency_mask]

    def __iter__(self):
        for _ in range(math.ceil(self.__len__() / self.batch_size)):
            current_batch_formation = defaultdict(int)
            for sample_size, d_id in zip(self.sample_sizes, self.rel_queries.sample(len(self.sample_sizes), weights=self.weights, replace=self.replace).keys()):
                current_batch_formation[d_id] += sample_size
            for d_id, sample_size in current_batch_formation.items():
                # q_ids = [q_id, *np.random.choice([p_id for p_id in self.rel_queries[d_id] if q_id != p_id], self.n_positives - 1, replace=False)]
                q_ids = [*np.random.choice(self.rel_queries[d_id], sample_size, replace=True)]
                for p_id in q_ids:
                    yield InputExample(texts=([self.queries[p_id]]), label=d_id)
