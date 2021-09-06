"""

"""
import math
import pickle
import pandas as pd
from torch.utils.data import IterableDataset
import random
import numpy as np

from . import RoundRobinRankingDataset
from ..readers import IRInputExample
from .util import pop_and_append
from sklearn.preprocessing import normalize


class RoundRobinTemplateRankingDataset(RoundRobinRankingDataset):
    def __init__(self, model, queries, responses, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, template_weight=0.15,
                 neg_rel_corpus=None):
        super().__init__(model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size, n_positives, temperature, shuffle, n_negatives, neg_rel_corpus)
        self.responses = pd.Series(responses)
        self.template_weight = template_weight

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.corpus.to_dict(), corpus=self.queries.to_dict(), rel_queries=self.neg_rel_corpus.to_dict())
        for batch_num in range(math.ceil(self.__len__() / self.batch_size)):
            for d_id, q_ids in self.rel_queries.sample(self.batch_size, weights=self.weights).map(self.retrieve_and_roll).items():
                d_mask = ~self.neg_rel_corpus.index.isin(self.rel_queries.get(d_id, []))
                weights = self.negatives_weighter(d_id)[d_mask]
                grouped_negatives = self.neg_rel_corpus[d_mask].rename('macro').map(lambda x: x[0]).to_frame().assign(w=weights)
                p = grouped_negatives.reset_index().groupby('macro').apply(lambda row: row.iloc[np.argmax(row.w)])
                n_ids = p.id.sample(self.n_negatives, weights=p.w)
                doc_text = self.responses[q_ids[0]] if random.random() > self.template_weight else self.corpus[d_id]

                yield IRInputExample(documents=([doc_text]), queries=([self.queries[q_id] for q_id in q_ids] + [self.queries[q_id] for q_id in n_ids]), label=batch_num, query_first=False)
