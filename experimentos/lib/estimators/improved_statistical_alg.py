from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL


class DigraphsMetrics:
    def __init__(self, average: float, std_dev: float, median: float):
        self.average = average
        self.std_dev = std_dev
        self.median = median

    average: float
    std_dev: float
    median: float


class ImprovedStatisticalAlg(BaseEstimator):
    threshold: float

    _digraphs_metrics: dict[str, DigraphsMetrics] = {}
    _training_data: pd.DataFrame = None

    def __init__(self, threshold: float = .7):
        self.threshold = threshold

    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {
            'threshold': self.threshold
        }

    def set_params(self, **params) -> Self:
        return ImprovedStatisticalAlg(params['threshold'])

    def fit(self, x: pd.DataFrame, y: list[int] | None = None):
        for digraph_name, values in x.items():
            average = values.mean()
            std_dev = values.std()
            median = values.median()
            self._digraphs_metrics[str(digraph_name)] = DigraphsMetrics(average, std_dev, median)
        self._training_data = x

    def predict(self, x: pd.DataFrame):
        results: list[int] = []
        for _, vec in x.iterrows():
            results.append(self.predict_vec(vec))
        return np.array(results)

    def predict_vec(self, x: pd.Series) -> int:
        hits = []
        for digraph_name, value in x.items():
            digraph_metrics = self._digraphs_metrics[str(digraph_name)]
            av = digraph_metrics.average
            std = digraph_metrics.std_dev
            md = digraph_metrics.median
            res = (min(av, md) * (0.95 - std / av)) <= value <= (max(av, md) * (1.05 + std / av))
            hits.append(res)
        length: int = len(hits)
        s: float = 1 if hits[0] else 0
        for i in range(1, length):
            if hits[i]:
                s += 1.5 if hits[i - 1] else 1
        result = GENUINE_LABEL if (s / ((length - 1) * 1.5 + 1)) >= self.threshold else IMPOSTOR_LABEL
        return result
