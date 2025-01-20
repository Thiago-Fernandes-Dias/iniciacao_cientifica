from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score 
from typing import Callable
from experimentos.cmu import *
from one_class_results import OneClassResults

class OneClassExperiment:
    cmu_database: CMUDataset
    estimator_factory: Callable[[], BaseEstimator]

    def __init__(self, cmu_database: CMUDataset, estimator_factory: Callable[[], BaseEstimator]) -> None:
        self.cmu_database = cmu_database
        self.estimator_factory = estimator_factory
    
    def exec(self) -> OneClassResults:
        X_training: dict[str, pd.DataFrame] = {}
        X_test: dict[str, pd.DataFrame] = {}
        y_training: dict[str, list[int]] = {}
        y_test: dict[str, list[int]] = {}
        one_class_estimators_map: dict[str, BaseEstimator] = {}

        for uk in self.cmu_database.user_keys():
            X_training[uk], y_training[uk] = self.cmu_database.one_vs_one_training_rows(uk)
            X_test[uk], y_test[uk] = self.cmu_database.one_vs_one_test_rows(uk)
    
        for uk in self.cmu_database.user_keys():
            one_class_estimators_map[uk] = self.estimator_factory().fit(X_training[uk], y_training[uk])

        user_model_acc_on_genuine_samples_map: dict[str, float] = {}
        user_model_recall_map: dict[str, float] = {}
        predictions_on_genuine_samples_map: dict[str, list[int]] = {}
        for uk in self.cmu_database.user_keys():
            predictions_on_genuine_samples_map[uk] = one_class_estimators_map[uk].predict(X_test[uk]).flatten().tolist()
            user_model_acc_on_genuine_samples_map[uk] = accuracy_score(y_test[uk], predictions_on_genuine_samples_map[uk])
            user_model_recall_map[uk] = recall_score(y_test[uk], predictions_on_genuine_samples_map[uk], average='micro')

        user_model_tn_rate_on_attack_samples_map: dict[str, float] = {}
        predictions_on_attacks_samples_map: dict[str, float] = {}
        for uk in self.cmu_database.user_keys():
            X_attacks, y_attacks = self.cmu_database.one_vs_one_attacks_rows(uk)
            predictions_on_attacks_samples_map[uk] = one_class_estimators_map[uk].predict(X_attacks).flatten().tolist()
            user_model_tn_rate_on_attack_samples_map[uk] = accuracy_score(y_attacks, predictions_on_attacks_samples_map[uk])

        results = OneClassResults(user_model_acc_on_genuine_samples_map, 
                                  user_model_recall_map, 
                                  user_model_tn_rate_on_attack_samples_map,
                                  predictions_on_genuine_samples_map,
                                  predictions_on_attacks_samples_map)

        return results
