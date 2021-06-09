"""

"""
import pickle
import pandas as pd
from torch.utils.data import IterableDataset

from ..readers import IRInputExample


class InformationRetrievalDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, weighter, batch_size=32, shuffle=True, n_negatives=1):
        self.model = model
        self.queries = pd.Series(queries).sample(frac=1)
        self.corpus = pd.Series(corpus)
        self.rel_queries = pickle.loads(pickle.dumps(pd.Series(rel_queries)))  # dirty copy
        self.rel_corpus = pickle.loads(pickle.dumps(pd.Series(rel_corpus)))  # dirty copy
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.weighter = weighter

    def __iter__(self):
        self.weighter.setup(self.model, queries=self.queries.to_dict(), corpus=self.corpus.to_dict(), rel_queries=self.rel_queries.to_dict())
        for i, (q_id, q_text) in enumerate(self.queries.items()):
            d_id = list(self.rel_corpus[q_id])[0]
            d_mask = self.rel_queries.index != d_id
            n_ids = self.rel_queries[d_mask].sample(self.n_negatives, weights=self.weighter(q_id)[d_mask]).keys()
            yield IRInputExample(texts=([q_text, self.corpus[d_id]] + [self.corpus[n_id] for n_id in n_ids]), label=i)

    def __len__(self):
        return len(self.queries)