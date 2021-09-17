import logging
import os
import shutil
from typing import List, Dict, Union
from numpy import ndarray
import torch
from torch import Tensor

from sentence_transformers.util import batch_to_device
from sentence_transformers import SentenceTransformer

QUERY_ENCODER_FOLDER = 'query_encoder'
DOCUMENT_ENCODER_FOLDER = 'document_encoder'

logger = logging.getLogger(__name__)


class BiSentenceTransformer(SentenceTransformer):
    """
    Loads or creates a bi-encoder architecture from 2 SentenceTransformer models, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    """

    def __init__(self, model_name_or_path: str = None, query_encoder: SentenceTransformer = None, document_encoder: SentenceTransformer = None, device: str = None):
        save_model_to = None
        logger.info("Load pretrained BiSentenceTransformer: {}".format(model_name_or_path))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        if model_name_or_path is not None and model_name_or_path != "":
            model_path = model_name_or_path

            #### Load from disk
            logger.info("Load BiSentenceTransformer from folder: {}".format(model_path))

            query_path = os.path.join(model_path, QUERY_ENCODER_FOLDER)
            if os.path.exists(query_path):
                logger.info("Loading query_encoder from folder: {}".format(query_path))
                query_encoder = SentenceTransformer(query_path, device=device)

            document_path = os.path.join(model_path, DOCUMENT_ENCODER_FOLDER)
            if os.path.exists(document_path):
                logger.info("Loading document_encoder from folder: {}".format(document_path))
                document_encoder = SentenceTransformer(document_path, device=device)

        super(SentenceTransformer, self).__init__()
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self._target_device = torch.device(device)

        # We created a new model from scratch based on a Transformer model. Save the SBERT model in the cache folder
        if save_model_to is not None:
            self.save(save_model_to)

    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False,
               encoder: str = 'query') -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :param encoder: The encoder to use

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """

        return self._get_encoder(encoder).encode(sentences=sentences,
                                                 batch_size=batch_size,
                                                 show_progress_bar=show_progress_bar,
                                                 output_value=output_value,
                                                 convert_to_numpy=convert_to_numpy,
                                                 convert_to_tensor=convert_to_tensor,
                                                 device=device,
                                                 normalize_embeddings=normalize_embeddings)

    def forward(self, sentence_feature: Dict[str, Tensor]):
        return self._get_encoder(sentence_feature['encoder'])(sentence_feature)

    def start_multi_process_pool(self, target_devices: List[str] = None):
        raise NotImplementedError

    @staticmethod
    def stop_multi_process_pool(pool):
        raise NotImplementedError

    def encode_multi_process(self, sentences: List[str], pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
        raise NotImplementedError

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        raise NotImplementedError

    def get_max_seq_length(self, encoder='query'):
        """
        Returns the maximal sequence length for input the model accepts. Longer inputs will be truncated
        """
        if hasattr(self._get_encoder(encoder)._first_module(), 'max_seq_length'):
            return self._get_encoder(encoder)._first_module().max_seq_length

        return None

    def tokenize(self, text: str, encoder='query'):
        """
        Tokenizes the text
        """
        return self._get_encoder(encoder)._first_module().tokenize(text)

    def get_sentence_features(self, *features, encoder='query'):
        return self._get_encoder(encoder)._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        raise NotImplementedError

    def _get_encoder(self, encoder_str):
        return self.query_encoder if encoder_str == 'query' else self.document_encoder

    def _first_module(self, encoder='query'):
        """Returns the first module of this sequential embedder"""
        q_encoder = self._get_encoder(encoder)
        return q_encoder._modules[next(iter(q_encoder._modules))]

    def _last_module(self, encoder='query'):
        """Returns the last module of this sequential embedder"""
        q_encoder = self._get_encoder(encoder)
        return q_encoder._modules[next(reversed(q_encoder._modules))]

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """
        if path is None:
            return

        self.query_encoder.save(os.path.join(path, QUERY_ENCODER_FOLDER))
        self.document_encoder.save(os.path.join(path, DOCUMENT_ENCODER_FOLDER))

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        encoders_mask = batch[0].encoder_mask
        texts = [[] for _ in range(num_texts)]
        ranking_labels = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            for idx, ranking_label in enumerate(example.labels):
                ranking_labels[idx].append(ranking_label)
            labels.append(example.label)

        if batch[0].labels:
            labels = ranking_labels
        labels = torch.tensor(labels).to(self._target_device)
        sentence_features = []

        for idx, encoder in enumerate(encoders_mask):
            tokenized = self.tokenize(texts[idx], encoder)
            tokenized['encoder'] = encoder
            batch_to_device(tokenized, self._target_device)
            sentence_features.append(tokenized)

        return sentence_features, labels

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step):
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({'step': int(subdir), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['step'])
                shutil.rmtree(old_checkpoints[0]['path'])

    @property
    def tokenizer(self, encoder='query'):
        """
        Property to get the tokenizer that is used by this model
        """
        return self._get_encoder(encoder)._first_module().tokenizer

    @tokenizer.setter
    def tokenizer(self, value, encoder='query'):
        """
        Property to set the tokenizer that should used by this model
        """
        self._get_encoder(encoder)._first_module().tokenizer = value

    @property
    def max_seq_length(self, encoder='query'):
        """
        Property to get the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        return self._get_encoder(encoder)._first_module().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value, encoder='query'):
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        self._get_encoder(encoder)._first_module().max_seq_length = value
