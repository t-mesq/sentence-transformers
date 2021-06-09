"""

"""
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset
from .util import pop_and_append
from ..readers import IRInputExample
from sklearn.preprocessing import normalize


class InformationRetrievalTemperatureDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, temperature=1, batch_size=32, shuffle=True, n_negatives=1):
        self.model = model
        self.queries = pd.Series(queries).sample(frac=1)
        self.corpus = pd.Series(corpus)
        self.rel_queries = pickle.loads(pickle.dumps(pd.Series(rel_queries)))  # dirty copy
        self.rel_corpus = pickle.loads(pickle.dumps(pd.Series(rel_corpus)))  # dirty copy
        self.batch_size = batch_size
        self.temperature_power = 1 / temperature
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.negatives_weighter = negatives_weighter
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.queries.to_dict(), corpus=self.corpus.to_dict(), rel_queries=self.rel_queries.to_dict())
        for i in range(self.__len__()):
            q_id = pop_and_append(self.rel_queries.sample(1, weights=self.weights).iloc[0])
            q_text = self.queries[q_id]
            d_id = list(self.rel_corpus[q_id])[0]
            d_mask = self.rel_queries.index != d_id
            n_ids = self.rel_queries[d_mask].sample(self.n_negatives, weights=self.negatives_weighter(q_id)[d_mask]).keys()
            yield IRInputExample(texts=([q_text, self.corpus[d_id]] + [self.corpus[n_id] for n_id in n_ids]), label=i)

    def __len__(self):
        return len(self.queries)