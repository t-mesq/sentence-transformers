from torch import Tensor

from . import InformationRetrievalWeigther
from ... import BiSentenceTransformer
from sentence_transformers.util import cos_sim, semantic_search
import numpy as np
from typing import Dict, Callable


class RTANCEWeighter(InformationRetrievalWeigther):
    """
    This class returns the similarity weights with each document, for every query.
    """

    def __init__(self,
                 query_chunk_size: int = 100,
                 corpus_chunk_size: int = 50000,
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
                 frequency: int = 1
                 ):
        self.frequency = frequency * batch_size
        self.query_chunk_size = query_chunk_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.score_function = score_function
        self.corpus_embeddings = None
        self.counter = self.frequency
        self.model = None
        self.queries = None
        self.corpus = None
        self.rel_queries = None
        self.corpus_embeddings = None
        self.query_args, self.document_args = {}, {}



    def setup(self, model, queries: Dict[str, str], corpus: Dict[str, str], rel_queries: Dict[str, str], query_embeddings: Tensor = None, corpus_embeddings: Tensor = None):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.rel_queries = rel_queries
        self.corpus_embeddings = corpus_embeddings

        if isinstance(model, BiSentenceTransformer):
            self.query_args['encoder'] = 'query'
            self.document_args['encoder'] = 'document'

    def __call__(self, q_id):
        self.update_counter()
        query_embedding = self.model.encode(self.queries.get(q_id), show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True, **self.query_args)

        query_result_list = semantic_search(query_embeddings=query_embedding,
                                              corpus_embeddings=self.corpus_embeddings,
                                              query_chunk_size=self.query_chunk_size,
                                              corpus_chunk_size=self.corpus_chunk_size,
                                              top_k=len(self.corpus_embeddings))[0]
        return np.abs(np.array(list(map(lambda x: x['score'], sorted(query_result_list, key=lambda x: x['corpus_id'])))) + 1.0)

    def update_counter(self):
        assert self.counter <= self.frequency   # shouldn't happen if frequency is reset

        if self.counter == self.frequency:
            self.counter = 0
            # update corpus_embeddings
            self.corpus_embeddings = self.model.encode(list(map(self.corpus.get, self.rel_queries)), show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True, **self.document_args)

        self.counter += 1
