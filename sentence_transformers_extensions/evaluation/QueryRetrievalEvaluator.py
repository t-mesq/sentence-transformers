import logging
from typing import List, Dict, Set, Callable

import pandas as pd
from torch import Tensor

from sentence_transformers.util import cos_sim, dot_score
from .DocumentRetrievalEvaluator import DocumentRetrievalEvaluator

logger = logging.getLogger(__name__)


class QueryRetrievalEvaluator(DocumentRetrievalEvaluator):
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
                 corpus_chunk_size: int = 1000,
                 mrr_at_k: List[int] = [10, 1000],
                 recall_at_k: List[int] = [1, 3, 5, 10, 20, 50, 100, 200, 500],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {'cos_sim': cos_sim, 'dot_score': dot_score},  # Score function, higher=more similar
                 main_score_function: str = None,
                 main_score_metric: str = 'recall@3',
                 compute_macro_metrics: bool = True,
                 ):
        super().__init__(queries=queries,
                         corpus = corpus,
                         relevant_docs=relevant_docs,
                         corpus_chunk_size=corpus_chunk_size,
                         mrr_at_k=mrr_at_k,
                         recall_at_k=recall_at_k,
                         show_progress_bar=show_progress_bar,
                         batch_size=batch_size,
                         name=name,
                         write_csv=write_csv,
                         score_functions=score_functions,
                         main_score_function=main_score_function,
                         main_score_metric=main_score_metric,
                         compute_macro_metrics=compute_macro_metrics
                         )
        self.query_to_doc = query_to_doc
        self.corpus_ids = list(map(self.query_to_doc.get, self.corpus_ids))

    # def get_ids_from_hits(self, hits):
    #     return pd.unique([self.query_to_doc[hit['corpus_id']] for hit in hits])
