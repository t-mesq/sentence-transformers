from typing import Iterable


class StackedCallbacks:
    def __init__(self, callbacks : Iterable):
        self.callbacks = callbacks

    def __call__(self, metric_scores, epochs, steps):
        for callback in self.callbacks:
            callback(metric_scores, epochs, steps)

