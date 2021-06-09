from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch import Tensor
import logging
from tqdm import tqdm, trange
from sentence_transformers.util import cos_sim, dot_score
import os
import numpy as np
from typing import List, Tuple, Dict, Set, Callable
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsScore:
    def __init__(self, score: float, scores: dict):
        self.score: float = score
        self.scores: dict = scores

    def __repr__(self):
        return "{}(score={}, scores={})".format(self.__class__.__name__, self.score, self.scores)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self.score == other.score
        else:
            return self.score == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.score)

    def __lt__(self, other):
        if other.__class__ is self.__class__:
            return self.score < other.score
        else:
            return self.score < other

    def __le__(self, other):
        if other.__class__ is self.__class__:
            return self.score <= other.score
        else:
            return self.score <= other

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)


class QueryRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting, specifically, template retrieval.

    Given a set of queries and a template corpus set. It will retrieve for each query the top-k most similar templates. It measures
    Mean Reciprocal Rank (MRR) and Recall@k.
    """

    def __init__(self,
                 queries: Dict[str, str],  # qid => query
                 corpus: Dict[str, str],  # cid => doc
                 relevant_docs: Dict[str, Set[str]],  # qid => Set[cid]
                 query_to_doc: Dict[str, str],  # qid => cid
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 recall_at_k: List[int] = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {'cos_sim': cos_sim, 'dot_score': dot_score},  # Score function, higher=more similar
                 main_score_function: str = None,
                 main_score_metric: str = 'recall@3'
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.query_to_doc = query_to_doc
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.recall_at_k = recall_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        self.main_score_metric = [{'metric': f'{metric}@k', 'k': int(k)} for metric, k in [main_score_metric.split('@')]][0]

        if name:
            name = "_" + name

        self.csv_file: str = "Template-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in recall_at_k:
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, corpus_model=None, corpus_embeddings: Tensor = None) -> MetricsScore:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        if corpus_model is None:
            corpus_model = model

        logger.info("Template Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        max_k = max(max(self.mrr_at_k), max(self.recall_at_k))

        # Compute embedding for the queries
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        itr = range(0, len(self.corpus), self.corpus_chunk_size)

        if self.show_progress_bar:
            itr = tqdm(itr, desc='Corpus Chunks', leave=False)

        # Iterate over chunks of the corpus
        for corpus_start_idx in itr:
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                cos_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        queries_result_list[name][query_itr].append({'corpus_id': corpus_id, 'score': score})

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        # Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        # Output
        for name in self.score_function_names:
            logging.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.recall_at_k:
                    output_data.append(scores[name]['recall@k'][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            score = max([scores[name][self.main_score_metric['metric']][self.main_score_metric['k']] for name in self.score_function_names])
        else:
            score = scores[self.main_score_function][self.main_score_metric['metric']][self.main_score_metric['k']]
        return MetricsScore(score, scores)

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        recall_at_k = {k: [] for k in self.recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            top_hits = pd.unique([self.query_to_doc[hit['corpus_id']] for hit in top_hits])
            query_relevant_docs = self.relevant_docs[query_id]

            # Recall@k
            for k_val in self.recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit in query_relevant_docs:
                        num_correct += 1

                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

        # Compute averages
        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        return {'recall@k': recall_at_k, 'mrr@k': MRR}

    def output_scores(self, scores):
        for k in scores['recall@k']:
            logger.info("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k] * 100))

        for k in scores['mrr@k']:
            logger.info("MRR@{}: {:.4f}".format(k, scores['mrr@k'][k]))