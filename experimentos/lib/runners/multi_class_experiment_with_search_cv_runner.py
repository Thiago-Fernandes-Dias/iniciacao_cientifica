from sklearn.model_selection._search import BaseSearchCV

from lib.datasets.dataset import *
from lib.multiclass_results import *

class MultiClassExperimentWithSearchCVRunner:

    def __init__(self, dataset: Dataset, estimator: BaseSearchCV):
        self.cmu_database = dataset
        self.estimator = estimator
    
    def exec(self) -> MultiClassResults:
        X_training, y_training = self.cmu_database.multi_class_training_samples()
        X_test, y_test = self.cmu_database.multi_class_test_samples()
        self.estimator.fit(X_training, y_training)
        predictions = self.estimator.predict(X_test)
        return MultiClassResults(y_test.to_list(), predictions, self.estimator.best_params_)
