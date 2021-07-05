import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from torch.utils.data import IterableDataset
import sys

from IPython.display import display, clear_output
import pandas as pd
# import seaborn as sns
from sentence_transformers_extensions import BiSentenceTransformer
from sentence_transformers_extensions.callbacks import MetricsScoresPrinter
from sentence_transformers_extensions.datasets import QueryFreqencyWeigther, ANCEWeighter, RoundRobinRankingDataset, InformationRetrievalTemperatureDataset
from sentence_transformers_extensions.readers import IRInputExample
from sentence_transformers_extensions.evaluation import TemplateRetrievalEvaluator, StackedRetrievalEvaluators

from sentence_transformers_extensions.losses import MultiplePositivesAndNegativesRankingLoss, agg_in_batch_negatives, agg_unique#, NormalizedDiscountedCumulativeGainLoss, NLLAndNDCGLoss, NLLAndMAPLoss, MeanAveragePrecisionLoss
from sentence_transformers_extensions.losses import TransposedMultiplePositivesAndNegativesRankingLoss, BiMultiplePositivesAndNegativesRankingLoss
tqdm.pandas

DATASET_FOLDER = "SB-templates/"
COLLECTIONS_PATH = "./"
DATASET_PATH = f"{COLLECTIONS_PATH}{DATASET_FOLDER}queries/"
TEMPLATES_FOLDER = f"{COLLECTIONS_PATH}{DATASET_FOLDER}templates/"
QUERY_MACROS = f"{COLLECTIONS_PATH}{DATASET_FOLDER}query-macros.json"
FILE_NAME = 'doc.json'

USE_GPU = True
USE_BI_SBERT = False
FINETUNNED_MODELS = {"paraphrase-xlm-r-multilingual-v1", 'paraphrase-distilroberta-base-v1', 'distiluse-base-multilingual-cased-v1'}
MODELS_DIR = f'/content/drive/MyDrive/Data/IST/tese/models/{"bi-"*USE_BI_SBERT}sbert'
SPLITS = 'train', 'val', 'test'

hard_negatives_pooling = 'none'
temperature = 1
negatives = 4
BATCH_SIZE = 32
positives = 1
shuffle_batches = 'smart'
in_batch_negatives = True
loss_name = 'nll'
EPOCHS = 50

queries_splits_df = pd.Series({split: pd.read_json(f"{DATASET_PATH}{split}/{FILE_NAME}") for split in SPLITS})
templates_df = pd.read_json(f"{TEMPLATES_FOLDER}{FILE_NAME}")
queries = queries_splits_df.map(lambda df: df.set_index('id').query_text) #dicts in the format: query_id -> query. Stores all training queries
corpus = templates_df.set_index('id').query_text #dict in the format: passage_id -> passage. Stores all existent passages
rel_docs = queries_splits_df.map(lambda df: df.set_index('id').macro_id.map(lambda x: [x])) #dicts in the format: query_id -> set(macro_ids). Stores all training queries
rel_queries = queries_splits_df.train.groupby('macro_id').id.agg(list)
scale = 16

weighters = {
    'none': QueryFreqencyWeigther(temperature=16),
    'ANCE': ANCEWeighter()
}


def get_positive_pairs(train_df, model=None):
    return train_df.join(templates_df.set_index('id'), on='macro_id', rsuffix='_macro').apply(lambda row: InputExample(texts=[row.query_text, row.query_text_macro], label=1), axis=1).sample(frac=1)


def get_smart_pairs(train_df, model=None):
    return RoundRobinRankingDataset(model=model, queries=queries.train, corpus=corpus, rel_queries=rel_queries, rel_corpus=rel_docs.train, batch_size=BATCH_SIZE, n_positives=positives, shuffle=shuffle_batches,
                                    temperature=temperature, n_negatives=negatives, negatives_weighter=weighters[hard_negatives_pooling])


def get_ir_smart_pairs(train_df, model=None):
    return InformationRetrievalTemperatureDataset(model=model, queries=queries.train, corpus=corpus, rel_queries=rel_queries, rel_corpus=rel_docs.train,
                                                  negatives_weighter=weighters[hard_negatives_pooling], temperature=temperature, batch_size=BATCH_SIZE, shuffle=shuffle_batches, n_negatives=negatives)


def get_ir_queries_pairs(train_df, model=None):
    return InformationRetrievalTemperatureDataset(model=model, queries=corpus, corpus=queries.train, rel_queries=rel_docs.train, rel_corpus=rel_queries,
                                                  negatives_weighter=weighters[hard_negatives_pooling], temperature=temperature, batch_size=BATCH_SIZE, shuffle=shuffle_batches, n_negatives=negatives,
                                                  query_first=True)


def get_train_examples(in_batch_neg, hard_neg_pool, shuffle):
    if shuffle == 'smart':
        return get_smart_pairs
    if shuffle == 'ir-smart':
        return get_ir_smart_pairs
    if shuffle == 'ir-queries':
        return get_ir_queries_pairs
    return get_positive_pairs

model = SentenceTransformer('distilroberta-base')

ir_evaluators = {split: TemplateRetrievalEvaluator(queries[split].to_dict(), corpus.to_dict(), rel_docs[split].to_dict(), name=split, main_score_function='cos_sim', main_score_metric='mrr@10', show_progress_bar=True) for split in ('val',)}
# x = ir_evaluators['val'](model, output_path="", epoch= -1, steps = -1, corpus_model=None, corpus_embeddings= None)
stacked_evaluator = StackedRetrievalEvaluators('val', **ir_evaluators)

from torch import nn
# Create the evaluator that is called during training

# Get trainning examples
trainning_examples = get_train_examples(in_batch_neg=in_batch_negatives, hard_neg_pool=hard_negatives_pooling, shuffle=shuffle_batches)(queries_splits_df.train, model)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataloader = DataLoader(trainning_examples, shuffle=(shuffle_batches is True), batch_size=BATCH_SIZE)
# train_dataloader = DataLoader(train_dataset, batch_size=32)
train_losses = {'nll': lambda *args, **kwargs: MultiplePositivesAndNegativesRankingLoss(*args, **kwargs, positives=positives, scale=scale),
                # 'pair-nll': lambda *args, **kwargs: MultiplePositivesAndNegativesRankingLoss(*args, **kwargs, positives=positives, scale=scale, cross_entropy_loss=PairwiseNLLLoss()),
                't-nll': lambda *args, **kwargs: TransposedMultiplePositivesAndNegativesRankingLoss(*args, **kwargs, positives=positives, scale=scale),
                'bi-nll': lambda *args, **kwargs: BiMultiplePositivesAndNegativesRankingLoss(*args, **kwargs, positives=positives, scale=scale),
                # 'map': lambda *args, **kwargs: MeanAveragePrecisionLoss(*args, **kwargs, regularization_strength=REGULARIZATION_STRENGTH),
                # 'ndcg': lambda *args, **kwargs: NormalizedDiscountedCumulativeGainLoss(*args, **kwargs, regularization_strength=REGULARIZATION_STRENGTH),
                # 'nll+ndcg': lambda *args, **kwargs: NLLAndNDCGLoss(*args, **kwargs, positives=positives, regularization_strength=REGULARIZATION_STRENGTH),
                # 'nll+map': lambda *args, **kwargs: NLLAndMAPLoss(*args, **kwargs, positives=positives, regularization_strength=REGULARIZATION_STRENGTH)
                }
agg = agg_in_batch_negatives if in_batch_negatives else agg_unique
train_loss = train_losses[loss_name](model=model, agg_fct=agg)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=stacked_evaluator,
          epochs=EPOCHS,
          warmup_steps=500,
          optimizer_params={'lr': 5e-5},
          output_path=".",
          use_amp=False,
          callback=MetricsScoresPrinter(EPOCHS)
          )
print()