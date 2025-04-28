from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from lib.datasets.dataset import *
from lib.multiclass_results import *

class MultiClassExperimentRunner:
    cmu_database: Dataset

    def __init__(self, dataset: Dataset, estimator: BaseEstimator):
        self.cmu_database = dataset
        self.estimator = estimator
    
    def exec(self) -> MultiClassResults:
        x_training, y_training = self.cmu_database.multi_class_training_samples()
        x_test, y_test = self.cmu_database.multi_class_test_samples()
        self.estimator.fit(x_training, y_training)
        predictions = self.estimator.predict(x_test)
        return MultiClassResults(y_test.to_list(), predictions, self.estimator.get_params(deep=False))
