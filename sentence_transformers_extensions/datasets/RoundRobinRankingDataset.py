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


class RoundRobinRankingDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, neg_rel_corpus=None):
        self.rel_corpus = rel_corpus
        self.negatives_weighter = negatives_weighter
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.rel_queries = pickle.loads(pickle.dumps(rel_queries))  # dirty copy
        self.neg_rel_corpus = self.rel_corpus if neg_rel_corpus is None else pd.Series(neg_rel_corpus)
        self.batch_size = batch_size
        self.n_positives = n_positives
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.temperature_power = 1 / temperature
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.corpus.to_dict(), corpus=self.queries.to_dict(), rel_queries=self.neg_rel_corpus.to_dict())
        for batch_num in range(math.ceil(self.__len__() / self.batch_size)):
            for d_id, q_ids in self.rel_queries.sample(self.batch_size, weights=self.weights).map(self.retrieve_and_roll).items():
                d_mask = ~self.neg_rel_corpus.index.isin(self.rel_queries[d_id])
                n_ids = self.neg_rel_corpus[d_mask].sample(self.n_negatives, weights=self.negatives_weighter(d_id)[d_mask]).keys()
                yield IRInputExample(texts=([self.corpus[d_id]] + [self.queries[q_id] for q_id in q_ids] + [self.queries[q_id] for q_id in n_ids]), label=batch_num, query_first=False)

    def __len__(self):
        return len(self.queries) // self.n_positives

    def retrieve_and_roll(self, x):
        return [pop_and_append(x) for _ in range(self.n_positives)]