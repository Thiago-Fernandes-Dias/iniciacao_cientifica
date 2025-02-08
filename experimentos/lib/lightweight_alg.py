from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class DigraphsMetrics:
    def __init__(self, average: float, std_dev: float, median: float):
        self.average = average
        self.std_dev = std_dev
        self.median = median

    average: float
    std_dev: float
    median: float


class LightWeightAlg(BaseEstimator):
    threshold: float

    _digraphs_metrics: dict[str, DigraphsMetrics] = {}

    def __init__(self, threshold: float = .7):
        self.threshold = threshold

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {
            'threshold': self.threshold
        }

    def set_params(self, **params) -> Self:
        return LightWeightAlg(params['threshold'])

    def fit(self, x: pd.DataFrame, y: list[int] | None = None):
        for digraph_name, values in x.items():
            average = values.mean()
            std_dev = values.std()
            median = values.median()
            self._digraphs_metrics[digraph_name] = DigraphsMetrics(average, std_dev, median)

    def predict(self, x: pd.DataFrame):
        digraphs_hits_matrix: list[list[bool]] = []
        results: list[int] = []

        for _, vec in x.iterrows():
            digraphs_hits_vec = []
            for digraph_name, value in vec.items():
                digraph_metrics = self._digraphs_metrics[digraph_name]
                av = digraph_metrics.average
                std = digraph_metrics.std_dev
                md = digraph_metrics.median
                res = (min(av, md) * (0.95 - std / av) <= value) and (value <= max(av, md) * (1.05 + std / av))
                digraphs_hits_vec.append(res)
            digraphs_hits_matrix.append(digraphs_hits_vec)

        l = len(digraphs_hits_matrix[0])
        for pred in digraphs_hits_matrix:
            s: float = 1 if pred[0] else 0
            for i in range(1, l):
                s += 1.5 if pred[i - 1] else 1
            results.append(1 if s / ((l - 1) * 1.5 + 1) >= 0.7 else -1)

        return np.array(results)
