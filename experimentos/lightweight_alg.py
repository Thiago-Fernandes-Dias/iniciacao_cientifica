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
    param1: int
    param2: int

    digraphs_metrics: dict[str, DigraphsMetrics]

    def __init__(self, param1: int = 1, param2: int = 2):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X: pd.DataFrame, y: list[int] | None = None):
        for digraph_name, values in X.items():
            average = values.mean() 
            std_dev = values.std()
            median = values.median()
            self.digraphs_metrics[digraph_name] = DigraphsMetrics(average, std_dev, median)
        time_spend_per_sampe = []
        for row in X.iterrows():
            time_spend_per_sampe.append(row[1].sum())

    def predict(self, X):
        return [self.param1 * x + self.param2 for x in X]