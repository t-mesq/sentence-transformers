import random
import pickle
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
from math import ceil
from collections import defaultdict
from ..readers import IRInputExample
from sklearn.preprocessing import normalize

from sentence_transformers import InputExample


class QuantileQuerySimilarityDataset(IterableDataset):
    def __init__(self, model, queries, rel_queries, rel_corpus, batch_size=32, quantile=4, temperature=1, n_positives=1):
        self.n_positives = n_positives
        self.quantile = quantile
        self.rel_corpus = pd.Series(rel_corpus)
        self.model = model
        self.queries = pd.Series(queries)
        self.rel_queries = pickle.loads(pickle.dumps(rel_queries))  # dirty copy
        self.batch_size = batch_size
        self.quantile_splits = self.get_quantile_splits()
        ibin = bin(self.batch_size)
        self.batch_formation = [int(ibin[:i], base=2) + int(ibin[i]) for i in range(-self.quantile, 0)]
        self.batch_formation = [self.batch_size - sum(self.batch_formation), *self.batch_formation]
        self.temperature_power = 1 / temperature
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])

    def __iter__(self):
        current_batch_formation = defaultdict(int)
        for i, total_s in enumerate(self.batch_formation[1:]):
            queries_mask = self.rel_queries.index.isin(self.quantile_splits[i])
            macro_ids = self.rel_queries[queries_mask].sample(self.n_positives, weights=self.weights[queries_mask], replace=True).index
            for macro_id, s in zip(macro_ids, self.get_even_split(total_s)):
                current_batch_formation[macro_id] += s

        current_batch = list(self.rel_corpus[self.rel_corpus.map(lambda x: x[0]).isin(self.quantile_splits[0])].sample(self.batch_formation[0]).keys())
        for macro_id, s in current_batch_formation.items():
            current_batch.extend(np.random.choice(self.rel_queries[macro_id], s))
        for id in current_batch:
            yield IRInputExample(texts=([self.queries[id]]), label=self.rel_corpus[id][0], query_first=True)

    def __len__(self):
        return len(self.queries)

    def get_quantile_splits(self):
        frequency_df = self.rel_queries.map(len).rename('counts').sort_values()
        frequency_df = self.rel_corpus.map(lambda y: y[0]).rename('macro').to_frame().join(frequency_df, on='macro').sort_values('counts')
        quantiles = frequency_df.quantile(np.arange(0, 1.00001, 0.25))
        quantiles.loc[0.00] = 0
        quantile_splits = zip(quantiles.counts, quantiles.counts.iloc[1:])
        return [frequency_df[(s1 <= frequency_df.counts) & (frequency_df.counts <= s2)].macro.unique() for s1, s2 in quantile_splits]

    def get_even_split(self, n):
        return [n // self.n_positives + bool(i < n % self.n_positives) for i in range(self.n_positives)]
