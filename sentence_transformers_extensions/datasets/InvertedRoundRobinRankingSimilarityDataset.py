"""

"""
import math
import pickle
import random

import pandas as pd
from torch.utils.data import  IterableDataset
import numpy as np
from ..readers import IRInputExample
from .util import pop_and_append
from sklearn.preprocessing import normalize


class InvertedRoundRobinRankingSimilarityDataset(IterableDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=2, temperature=1, shuffle=True, n_negatives=0, neg_rel_queries=None, top_k_sampling=False, replace=True, random_p_sampling=True, responses=None, template_weight=0.15):
        self.template_weight = template_weight
        self.random_p_sampling = self.retrieve_random if random_p_sampling else self.retrieve_and_roll
        self.replace = replace
        self.top_k_sampling = top_k_sampling
        self.rel_corpus = rel_corpus
        self.negatives_weighter = negatives_weighter
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.rel_queries = pd.Series(pickle.loads(pickle.dumps(rel_queries)))  # dirty copy
        self.neg_rel_queries = self.rel_queries if neg_rel_queries is None else pd.Series(neg_rel_queries)
        self.batch_size = batch_size
        self.n_positives = n_positives
        self.shuffle = shuffle
        self.n_negatives = n_negatives
        self.temperature_power = 1 / temperature
        self.weights = self.rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])
        self.negative_sample_sizes = list(map(len, np.array_split(np.ones(self.n_negatives).astype(int), self.n_positives)))
        self.responses = None if responses is None else pd.Series(responses)

    def __iter__(self):
        self.negatives_weighter.setup(self.model, queries=self.queries.to_dict(), corpus=self.corpus.to_dict(), rel_queries=self.neg_rel_queries.to_dict())
        for batch_num in range(math.ceil(self.__len__() / self.batch_size)):
            sampled_ids = self.rel_queries.sample(self.batch_size, weights=self.weights, replace=self.replace)
            available_docs = set(sampled_ids.keys())
            for d_id, q_ids in self.positives_sample_generator(sampled_ids, available_docs):
                labels = [*map(lambda x: self.rel_corpus.get(x)[0], q_ids), d_id]     # n_positives queries + positive document
                available_docs.add(d_id)

                n_ids = []
                for q_id, sample_size in zip(q_ids, self.negative_sample_sizes):
                    if sample_size == 0:
                        break

                    d_mask = ~self.neg_rel_queries.index.isin(available_docs)
                    if self.top_k_sampling:
                        x = self.negatives_weighter(q_id)[d_mask]
                        n_ids.extend(self.neg_rel_queries[d_mask].iloc[x.argsort()[-sample_size:]].keys())
                    else:
                        n_ids.extend(self.neg_rel_queries[d_mask].sample(sample_size, weights=self.negatives_weighter(q_id)[d_mask]).keys())
                    available_docs.update(n_ids)

                labels.extend(n_ids)
                yield IRInputExample(queries=[self.queries[q_id] for q_id in q_ids], documents=([self.get_document_text[d_id]] + [self.get_document_text[d_id] for d_id in n_ids]), label=batch_num, query_first=True, labels=labels)

    def __len__(self):
        return len(self.queries) // self.n_positives

    def get_document_text(self, d_id):
        if self.responses is None:
            return self.self.corpus[d_id]
        return self.responses[random.sample(self.rel_queries[d_id], 1)[0]] if random.random() > self.template_weight else self.corpus[d_id]

    def positives_sample_generator(self, d_ids, available_docs):
        u_ids = dict(zip(*np.unique(d_ids.keys(), return_counts=True)))
        extended_mask = ~self.rel_queries.index.isin(u_ids)
        extended_ids = self.neg_rel_queries[extended_mask].sample(len(d_ids) - len(u_ids)).keys()
        u_ids.update(dict(zip(extended_ids, [0]*len(extended_ids))))
        q_ids = []
        for d_id in u_ids:
            q_ids.extend(self.random_p_sampling(self.rel_queries.loc[d_id], self.n_positives*u_ids[d_id]))
        grouped_q_ids = np.array(q_ids).reshape((-1, self.n_positives))
        return dict(zip(u_ids, grouped_q_ids)).items()

    @staticmethod
    def retrieve_and_roll(x, n):
        return [pop_and_append(x) for _ in range(n)]

    @staticmethod
    def retrieve_random(x, n):
        return np.random.choice(x, n, replace=True)


class InvertedRoundRobinBalancedRankingSimilarityDataset(InvertedRoundRobinRankingSimilarityDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=1, temperature=1, shuffle=True, n_negatives=0, neg_rel_queries=None, top_k_sampling=False, random_p_sampling=True, responses=None, template_weight=0.15):
        super().__init__(model=model, queries=queries, corpus=corpus, rel_queries=rel_queries, rel_corpus=rel_corpus, negatives_weighter=negatives_weighter, batch_size=batch_size, n_positives=n_positives, temperature=temperature, shuffle=shuffle, n_negatives=n_negatives, neg_rel_queries=neg_rel_queries, top_k_sampling=top_k_sampling, random_p_sampling=random_p_sampling, replace=False, responses=responses, template_weight=template_weight)
        self.query_weights = self.weights
        self.weights = np.ones_like(self.rel_queries)
        self.counts = self.rel_queries.map(len)

    def positives_sample_generator(self, d_ids, available_docs):
        mask = self.rel_queries.index.isin(d_ids.keys())
        query_counts = pd.Series(dict(zip(*np.unique(self.rel_queries[mask].sample(self.n_positives * len(d_ids), weights=self.query_weights[mask], replace=True).keys(), return_counts=True))))
        q_ids = []
        for d_id, n in query_counts.items():
            q_ids.extend(self.random_p_sampling(self.rel_queries.loc[d_id], n))
        grouped_q_ids = np.array(q_ids).reshape((-1, self.n_positives))
        return dict(zip(d_ids.keys(), grouped_q_ids)).items()


class ExpandedInvertedRoundRobinBalancedRankingSimilarityDataset(InvertedRoundRobinBalancedRankingSimilarityDataset):
    def __init__(self, model, queries, corpus, rel_queries, rel_corpus, negatives_weighter, batch_size=32, n_positives=1, temperature=1, shuffle=True, n_negatives=0, neg_rel_queries=None, top_k_sampling=False, random_p_sampling=True, expanded_prob=0.1, responses=None, template_weight=0.15):
        super().__init__(model=model, queries=queries, corpus=corpus, rel_queries=rel_queries, rel_corpus=rel_corpus, negatives_weighter=negatives_weighter, batch_size=batch_size, n_positives=n_positives, temperature=temperature, shuffle=shuffle, n_negatives=n_negatives, neg_rel_queries=neg_rel_queries, top_k_sampling=top_k_sampling, random_p_sampling=random_p_sampling, responses=responses, template_weight=template_weight)
        self.expanded_prob = expanded_prob
        self.neg_queries = self.rel_queries[~self.rel_queries.index.isin(self.neg_rel_queries.index)]
        self.expansion_size = int(self.batch_size * expanded_prob)
        self.expansion_size_p = self.batch_size * expanded_prob - self.expansion_size
        self.neg_rel_weights = self.neg_rel_queries.map(len).agg(lambda x: normalize(np.array([x]) ** self.temperature_power, norm='l1')[0])

    def positives_sample_generator(self, d_ids, available_docs):
        current_expansion_size = self.expansion_size + (np.random.rand() < self.expansion_size_p)
        d_ids = self.neg_rel_queries.sample(self.batch_size - current_expansion_size)
        available_docs.clear()
        available_docs.update(set(d_ids.keys()))
        mask = self.neg_rel_queries.index.isin(d_ids.keys())
        query_counts = pd.Series(dict(zip(*np.unique(self.neg_rel_queries[mask].sample(self.n_positives * self.batch_size - current_expansion_size, weights=self.neg_rel_weights[mask], replace=True).keys(), return_counts=True))))
        expanded_ids = self.neg_queries.sample(current_expansion_size).map(lambda x: 1)
        query_counts = query_counts.append(expanded_ids)
        q_ids = []
        for d_id, n in query_counts.items():
            q_ids.extend(self.random_p_sampling(self.rel_queries.loc[d_id], n))
        grouped_q_ids = np.array(q_ids).reshape((-1, self.n_positives))
        return dict(zip([*d_ids.keys(), *expanded_ids.keys()], grouped_q_ids)).items()
