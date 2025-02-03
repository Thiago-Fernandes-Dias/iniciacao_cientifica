import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from typing import Self

class DigraphsMetrics:
    def __init__(self, average: float, std_dev: float, median: float):
        self.average = average
        self.std_dev = std_dev
        self.median = median
    average: float
    std_dev: float
    median: float

class LightWeightAlg(BaseEstimator):
    a: float
    b: float
    threshold: float

    _digraphs_metrics: dict[str, DigraphsMetrics] = {}

    def __init__(self, a: float = .95, b: float = 1.05, threshold: float = .7):
        self.a = a
        self.b = b
        self.threshold = threshold
    
    def get_params(self, deep: bool = True) -> dict[str, float]:
        return {
            'a': self.a,
            'b': self.b,
            'threshold': self.threshold
        }
    
    def set_params(self, **params) -> Self:
        return LightWeightAlg(params['a'], params['b'], params['threshold'])

    def fit(self, X: pd.DataFrame, y: list[int] | None = None):
        for digraph_name, values in X.items():
            average = values.mean() 
            std_dev = values.std()
            median = values.median()
            self._digraphs_metrics[digraph_name] = DigraphsMetrics(average, std_dev, median)
        time_spend_per_sample = []
        for row in X.iterrows():
            time_spend_per_sample.append(row[1].sum())
        self.is_fitted = True

    def predict(self, X: pd.DataFrame):
        digraphs_hits_matrix: list[list[bool]] = []
        results: list[int] = []
        
        for _, vec in X.iterrows():
            digraphs_hits_vec = []
            for digraph_name, value in vec.items():
                digraph_metrics = self._digraphs_metrics[digraph_name]
                av = digraph_metrics.average
                std = digraph_metrics.std_dev
                md = digraph_metrics.median
                res = min(av, md) * (self.a - std / av) <= value 
                res = res and value <= max(av, md) * (self.b + std / av)
                digraphs_hits_vec.append(res)
            digraphs_hits_matrix.append(digraphs_hits_vec)

        for pred in digraphs_hits_matrix:
            s: float = 1
            l = len(pred)
            for i in range(2, l):
                s += 1.5 if pred[i - 1] else 1
            results.append(1 if s / (9 * 1.5 + 1) >= 0.7 else -1) 
        
        return np.array(results)


