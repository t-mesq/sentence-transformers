"""

"""
import math
import pickle
from torch.utils.data import  IterableDataset
import numpy as np
from ..readers import IRInputExample
from .util import pop_and_append
from sklearn.preprocessing import normalize


class RoundRobinRankingDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.rel_queries = pickle.loads(pickle.dumps(rel_queries))  # dirty copy
        self.batch_size = batch_size
        self.n_positives = n_positives
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.temperature_power = 1 / temperature
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])

    def __iter__(self):
        for batch_num in range(math.ceil(self.__len__() / self.batch_size)):
            for d_id, q_ids in self.rel_queries.sample(self.batch_size, weights=self.weights).map(self.retrieve_and_roll).items():
                d_mask = self.rel_queries.index != d_id
                n_ids = [pop_and_append(self.rel_queries[d_mask].sample(1, weights=self.weights[d_mask]).iloc[0]) for _ in range(self.n_negatives)]
                yield IRInputExample(texts=([self.corpus[d_id]] + [self.queries[q_id] for q_id in q_ids] + [self.queries[q_id] for q_id in n_ids]), label=batch_num, query_first=False)

    def __len__(self):
        return len(self.queries) // self.n_positives

    def retrieve_and_roll(self, x):
        return [pop_and_append(x) for _ in range(self.n_positives)]