"""

"""
import math

import numpy as np

from sentence_transformers import InputExample
from . import RoundRobinRankingDataset
from .util import pop_and_append


class RoundRobinQueryRankingDataset(RoundRobinRankingDataset):
    def __init__(self, model, queries, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, neg_rel_corpus=None):
        super().__init__(model, queries, None, rel_queries, rel_corpus, negatives_weighter, batch_size, n_positives, temperature, shuffle, n_negatives, neg_rel_corpus)

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.queries.to_dict(), corpus=self.queries.to_dict(), rel_queries=self.neg_rel_corpus.to_dict())
        for batch_num in range(math.ceil(self.__len__() / self.batch_size)):
            for d_id, q_id in self.rel_queries.sample(self.batch_size, weights=self.weights).map(pop_and_append).items():
                d_mask = ~self.neg_rel_corpus.index.isin(self.rel_queries[d_id])
                n_ids = self.neg_rel_corpus[d_mask].sample(self.n_negatives, weights=self.negatives_weighter(d_id)[d_mask]).keys()
                q_ids = np.random.choice(self.rel_queries[d_id], self.n_positives)
                yield InputExample(texts=([self.queries[p_id] for p_id in [q_id, *q_ids]] + [self.queries[q_id] for q_id in n_ids]), label=batch_num)
