"""

"""
import math
import pickle
import pandas as pd
from torch.utils.data import  IterableDataset
import numpy as np
from ..readers import IRInputExample
from .util import pop_and_append
from sklearn.preprocessing import normalize


class InvertedRoundRobinRankingDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, neg_rel_queries=None):
        self.rel_corpus = rel_corpus
        self.negatives_weighter = negatives_weighter
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.rel_queries = pickle.loads(pickle.dumps(rel_queries))  # dirty copy
        self.neg_rel_queries = self.rel_queries if neg_rel_queries is None else pd.Series(neg_rel_queries)
        self.batch_size = batch_size
        self.n_positives = n_positives
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.temperature_power = 1 / temperature
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.queries.to_dict(), corpus=self.corpus.to_dict(), rel_queries=self.neg_rel_queries.to_dict())
        for batch_num in range(math.ceil(self.__len__() / self.batch_size)):
            for d_id, q_ids in self.rel_queries.sample(self.batch_size, weights=self.weights).map(self.retrieve_and_roll).items():
                q_id = q_ids[0]
                d_mask = ~self.neg_rel_queries.index.isin(self.rel_corpus[q_id])
                n_ids = self.neg_rel_queries[d_mask].sample(self.n_negatives, weights=self.negatives_weighter(q_id)[d_mask]).keys()
                yield IRInputExample(queries=[self.queries[q_id]], documents=([self.corpus[d_id]] + [self.corpus[d_id] for d_id in n_ids]), label=batch_num, query_first=True)

    def __len__(self):
        return len(self.queries) // self.n_positives

    def retrieve_and_roll(self, x):
        return [pop_and_append(x) for _ in range(self.n_positives)]