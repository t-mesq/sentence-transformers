"""

"""
import math
from collections import defaultdict

import numpy as np
import numpy.random

from sentence_transformers import InputExample
from . import RoundRobinRankingDataset


class QueryExclusiveSimilarityDataset(RoundRobinRankingDataset):
    def __init__(self, model, queries, rel_queries, rel_corpus, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, neg_rel_corpus=None, replace=True):
        super().__init__(model, queries, None, rel_queries, rel_corpus, None, batch_size, n_positives, temperature, shuffle, n_negatives, neg_rel_corpus)
        self.replace = replace
        self.counts = self.rel_queries.map(len)

    def __iter__(self):
        for _ in range(math.ceil(self.__len__() / self.batch_size)):
            current_batch_formation = defaultdict(int)
            available_d_ids = set(self.rel_queries.keys())
            current_batch_size = 0

            while current_batch_size < self.batch_size:
                # sampling n=batch_size covers worst case with 1 positive and no repetitions
                d_sample = self.rel_queries.sample(self.batch_size, weights=self.weights, replace=self.replace).keys()
                d_sample = d_sample[d_sample.isin(available_d_ids)]
                for d_id in d_sample:
                    if current_batch_size >= self.batch_size:
                        break

                    if (current_batch_formation[d_id] < self.counts[d_id]) and (d_id in available_d_ids):
                        n_samples = min(self.n_positives, self.batch_size - current_batch_size, self.counts[d_id] - current_batch_formation[d_id])
                        current_batch_formation[d_id] += n_samples
                        current_batch_size += n_samples
                    else:
                        available_d_ids.remove(d_id)

            for d_id, sample_size in current_batch_formation.items():
                q_ids = [*np.random.choice(self.rel_queries[d_id], sample_size, replace=False)]
                for p_id in q_ids:
                    yield InputExample(texts=([self.queries[p_id]]), label=d_id)
