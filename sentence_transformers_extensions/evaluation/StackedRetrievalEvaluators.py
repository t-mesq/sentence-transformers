from. import DocumentRetrievalEvaluator, MetricsScore
from sentence_transformers.evaluation import SentenceEvaluator


class StackedRetrievalEvaluators(SentenceEvaluator):
    def __init__(self, tracked, **evaluators):
        self.tracked = tracked
        self.evaluators = evaluators

    def __call__(self, *kargs, **kwargs):
        metric_scores = {split: evaluator(*kargs, **kwargs) for split, evaluator in self.evaluators.items()}
        return MetricsScore(score=metric_scores[self.tracked].score, scores={split: ms.scores for split, ms in metric_scores.items()})