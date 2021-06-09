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
import seaborn as sns
from sentence_transformers_extensions import BiSentenceTransformer
from sentence_transformers_extensions.readers import IRInputExample
from sentence_transformers_extensions.evaluation import TemplateRetrievalEvaluator
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

queries_splits_df = pd.Series({split: pd.read_json(f"{DATASET_PATH}{split}/{FILE_NAME}") for split in SPLITS})
templates_df = pd.read_json(f"{TEMPLATES_FOLDER}{FILE_NAME}")
queries = queries_splits_df.map(lambda df: df.set_index('id').query_text) #dicts in the format: query_id -> query. Stores all training queries
corpus = templates_df.set_index('id').query_text #dict in the format: passage_id -> passage. Stores all existent passages
rel_docs = queries_splits_df.map(lambda df: df.set_index('id').macro_id.map(lambda x: {x})) #dicts in the format: query_id -> set(macro_ids). Stores all training queries
rel_queries = queries_splits_df.train.groupby('macro_id').id.agg(list)


model = SentenceTransformer('distilroberta-base')

ir_evaluators = {split: TemplateRetrievalEvaluator(queries[split].to_dict(), corpus.to_dict(), rel_docs[split].to_dict(), name=split, main_score_function='cos_sim', main_score_metric='mrr@10', show_progress_bar=True, bi_sbert=USE_BI_SBERT) for split in ('val',)}
x = ir_evaluators['val'](model, output_path="", epoch= -1, steps = -1, corpus_model=None, corpus_embeddings= None)
print()
print()