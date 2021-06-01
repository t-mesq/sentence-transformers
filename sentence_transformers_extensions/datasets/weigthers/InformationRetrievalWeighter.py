import numpy as np
from typing import Dict


class InformationRetrievalWeigther:
    """
    Base class for all Weighters

    Extend this class and implement __call__ and setup() for custom Weighters.
    """
    def setup(self, model, queries: Dict[str, str], corpus: Dict[str, str], rel_queries: Dict[str, str], **kwargs):
        pass

    def __call__(self, q_id, **kwargs) -> np.array:
        pass