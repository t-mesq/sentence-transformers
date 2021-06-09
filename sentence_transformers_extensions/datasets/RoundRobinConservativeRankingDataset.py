import random
import pickle
from torch.utils.data import IterableDataset
from sentence_transformers import InputExample


class RoundRobinConservativeRankingDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, batch_size=32, n_pairs=2, shuffle=True):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.rel_queries = rel_queries.copy()
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.shuffle = shuffle
        self.rr_map = self.get_round_robin_map()

    def __iter__(self):
        self.rr_map = self.get_round_robin_map()
        for d_id, q_ids in self.rr_map:
            yield InputExample(texts=([self.corpus[d_id]] + [self.queries[q_id] for q_id in q_ids]), label=1)

    def __len__(self):
        return len(self.rr_map)

    def get_round_robin_map(self):
        sampled_queries_map = []
        sampled_queries = pickle.loads(pickle.dumps(self.rel_queries))  # dirty copy
        if self.shuffle:
            sampled_queries.map(random.shuffle)
        while len(sampled_queries) >= self.batch_size:
            sampled_queries = sampled_queries[sampled_queries.map(len) >= self.n_pairs].sort_values(ascending=False, key=lambda s: s.map(len))
            sampled_queries_map.extend(sampled_queries.head(self.batch_size).map(lambda lst: [lst.pop() for _ in range(self.n_pairs)]).items())
        return sampled_queries_map