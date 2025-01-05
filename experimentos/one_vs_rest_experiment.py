from cmu import *

from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score, recall_score

import numpy as np

class OneVsRestExperiment:
    cmu_database: CMUDatabase
    estimator_factory: Callable[[], BaseEstimator]

    def __init__(self, cmu_database: CMUDatabase, estimator_factory: Callable[[], BaseEstimator]) -> None:
        self.cmu_database = cmu_database
        self.estimator_factory = estimator_factory
    
    def exec(self):
        X_training: dict[str, pd.DataFrame] = {}
        X_test: dict[str, pd.DataFrame] = {}
        y_training: dict[str, list[int]] = {}
        y_test: dict[str, list[int]] = {}
        estimators_map: dict[str, BaseEstimator] = {}

        for uk in self.cmu_database.user_keys():
            X_training[uk], y_training[uk] = self.cmu_database.one_vs_rest_training_rows(uk)
            X_test[uk], y_test[uk] = self.cmu_database.one_vs_rest_test_rows(uk)
        
        for uk in self.cmu_database.user_keys():
            estimators_map[uk] = self.estimator_factory().fit(X_training[uk], y_training[uk])
        
        two_class_acc_map: dict[str, float] = {}
        two_class_recall_map: dict[str, float] = {}

        for uk in self.cmu_database.user_keys():
            predictions = estimators_map[uk].predict(X_test[uk]).flatten().tolist()
            two_class_acc_map[uk] = balanced_accuracy_score(y_test[uk], predictions)
            two_class_recall_map[uk] = recall_score(y_test[uk], predictions, average='micro')

        average_acc = np.average(list(two_class_acc_map.values()))
        average_recall = np.average(list(two_class_recall_map.values()))

        print(f"Acurácia dos modelos One-Vs-One: {average_acc}")
        print(f"Recall dos modelos One-Vs-One: {average_recall}")