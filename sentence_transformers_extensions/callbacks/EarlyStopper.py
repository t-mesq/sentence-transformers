from sentence_transformers_extensions.evaluation import MetricsScore


class EarlyStoppingException(Exception):
    """Controlled exception raised when model stops early.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, patience, max_score):
        super().__init__(f'The model did not surpass the score {max_score:.3f}, after {patience} epochs')


class EarlyStopper:
    def __init__(self, patience: int, min_score: bool = False):
        self.patience = patience
        self.sign = -1 if min_score else 1
        self.epochs_since_max = 0
        self.max_score = -float('inf')

    def __call__(self, metric_scores: MetricsScore, epochs, steps):
        score = metric_scores.score * self.sign
        if self.max_score < score:
            self.max_score = score
            self.epochs_since_max = 0

        self.epochs_since_max += 1

        if self.epochs_since_max > self.patience:
            raise EarlyStoppingException(patience=self.patience, max_score=self.max_score*self.sign)
