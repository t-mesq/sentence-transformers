"""

"""
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import IterableDataset
from .util import pop_and_append
from ..readers import IRInputExample
from sklearn.preprocessing import normalize
import random


class InformationRetrievalWithTemplatesTemperatureDataset(IterableDataset):
    def __init__(self, model, queries, responses, templates, rel_queries, rel_corpus, negatives_weighter, temperature=1, batch_size=32, shuffle=True, n_negatives=1, query_first=True,
                 template_weight=0.15, neg_rel_queries=None):
        self.template_weight = template_weight
        self.responses = pd.Series(responses)
        self.templates = pd.Series(templates)
        self.model = model
        self.queries = pd.Series(queries).sample(frac=1)
        self.rel_queries = pickle.loads(pickle.dumps(pd.Series(rel_queries)))  # dirty copy
        self.rel_corpus = pickle.loads(pickle.dumps(pd.Series(rel_corpus)))  # dirty copy
        self.neg_rel_queries = self.rel_queries if neg_rel_queries is None else pd.Series(neg_rel_queries)
        self.batch_size = batch_size
        self.temperature_power = 1 / temperature
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.negatives_weighter = negatives_weighter
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])
        self.query_first = query_first

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.queries.to_dict(), corpus=self.templates.to_dict(), rel_queries=self.neg_rel_queries.to_dict())
        for i in range(self.__len__()):
            q_id = pop_and_append(self.rel_queries.sample(1, weights=self.weights).iloc[0])
            q_text = self.queries[q_id]

            d_id = pop_and_append(self.rel_corpus[q_id])
            d_text = self.responses[q_id] if random.random() > self.template_weight else self.templates[d_id]

            d_mask = ~self.neg_rel_queries.index.isin(self.rel_corpus.get(q_id, []))
            n_ids = self.neg_rel_queries[d_mask].sample(self.n_negatives, weights=self.negatives_weighter(q_id)[d_mask]).keys()
            n_ids_texts = [self.responses[random.sample(self.rel_queries[n_id], 1)] if random.random() > self.template_weight else self.templates[n_id] for n_id in n_ids]

            yield IRInputExample(texts=[q_text, d_text, *n_ids_texts], label=i, query_first=self.query_first)

    def __len__(self):
        return len(self.queries)