from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from cmu_dataset import *
from multiclass_results import *

class MultiClassExperiment:
    cmu_database: CMUDataset
    estimator: BaseEstimator

    def __init__(self, dataset: CMUDataset, estimator: BaseEstimator):
        self.cmu_database = dataset
        self.estimator = estimator
    
    def exec(self) -> MultiClassResults:
        X_training, y_training = self.cmu_database.multi_class_training_samples()
        X_test, y_test = self.cmu_database.multi_class_test_samples()
        self.estimator.fit(X_training, y_training)
        predictions = self.estimator.predict(X_test)
        return MultiClassResults(y_test, predictions, self.estimator.get_params(deep=False))
