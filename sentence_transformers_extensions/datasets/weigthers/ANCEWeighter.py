from torch import Tensor

from . import InformationRetrievalWeigther
from ... import BiSentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search
import numpy as np
from typing import Dict, Callable


class ANCEWeighter(InformationRetrievalWeigther):
    """
    This class returns the similarity weights with each document, for every query.
    """

    def __init__(self,
                 query_chunk_size: int = 100,
                 corpus_chunk_size: int = 50000,
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim
                 ):
        self.query_chunk_size = query_chunk_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.score_function = score_function
        self.queries_weights = None

    def setup(self, model, queries: Dict[str, str], corpus: Dict[str, str], rel_queries: Dict[str, str], query_embeddings: Tensor = None, corpus_embeddings: Tensor = None):

        query_args, document_args = {}, {}
        if isinstance(model, BiSentenceTransformer):
            query_args['encoder'] = 'query'
            document_args['encoder'] = 'document'

        if query_embeddings is None:
            # Compute embedding for the queries
            query_embeddings = model.encode(list(queries.values()), show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True, **query_args)

        if corpus_embeddings is None:
            # Compute embedding for the corpus
            corpus_embeddings = model.encode(list(map(corpus.get, rel_queries)), show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True, **document_args)

        queries_result_list = semantic_search(query_embeddings=query_embeddings,
                                              corpus_embeddings=corpus_embeddings,
                                              query_chunk_size=self.query_chunk_size,
                                              corpus_chunk_size=self.corpus_chunk_size,
                                              top_k=len(corpus_embeddings))

        self.queries_weights = {q_id: np.abs(np.array(list(map(lambda x: x['score'], sorted(scores, key=lambda x: x['corpus_id'])))) + 1.0) for q_id, scores in zip(queries, queries_result_list)}

    def __call__(self, q_id):
        return self.queries_weights[q_id]
