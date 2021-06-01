import numpy as np
import pandas as pd
from . import InformationRetrievalWeigther
from sklearn.preprocessing import normalize
from typing import Dict


class QueryFreqencyWeigther(InformationRetrievalWeigther):
    """
    This class returns the Query Frequency weights of each document, for every query.

    """
    def __init__(self, temperature=1):
        self.temperature_power = 1/temperature
        self.weights = None

    def setup(self, model, queries: Dict[str, str], corpus: Dict[str, str], rel_queries: Dict[str, str], **kwargs):
        self.weights = pd.Series(rel_queries).map(len).agg(lambda x: normalize(np.array([x])**self.temperature_power, norm='l1')[0])

    def __call__(self, q_id, **kwargs) -> np.array:
        return self.weights