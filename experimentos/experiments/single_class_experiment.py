from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score 
from typing import Callable
from cmu_dataset import *
from one_class_results import OneClassResults
from experiments.one_class_experiment import OneClassExperiment

class SingleClassExperiment(OneClassExperiment):
    _cmu_database: CMUDataset
    _estimator: BaseEstimator
    _get_estimator_params: Callable[[BaseEstimator], dict]

    def __init__(self, dataset: CMUDataset, estimator: BaseEstimator, get_estimator_params: Callable[[BaseEstimator], dict]) -> None:
        self._cmu_database = dataset
        self._estimator =estimator
        self._get_estimator_params = get_estimator_params
    
    def exec(self, *, include_hpo: bool = False) -> OneClassResults:
        user_model_acc_on_genuine_samples_map: dict[str, float] = {}
        user_model_recall_map: dict[str, float] = {}
        predictions_on_genuine_samples_map: dict[str, list[int]] = {}
        user_model_tn_rate_on_attack_samples_map: dict[str, float] = {}
        predictions_on_attacks_samples_map: dict[str, list[ int ]] = {}
        one_class_estimators_hp_map: dict[str, dict[str, object]] = {}

        one_class_estimators_hp_map['default'] = self._get_estimator_params(self._estimator)

        for uk in self._cmu_database.user_keys():
            X_training, y_training = self._cmu_database.one_vs_one_training_rows(uk)
            X_test, y_test = self._cmu_database.one_vs_one_test_rows(uk)
            X_attacks, y_attacks = self._cmu_database.one_vs_one_attacks_rows(uk)
            self._estimator.fit(X_training, y_training)
            if (include_hpo):
                one_class_estimators_hp_map[uk] = self._estimator.get_params()
            predictions_on_genuine_samples_map[uk] = self._estimator.predict(X_test).flatten().tolist()
            user_model_acc_on_genuine_samples_map[uk] = accuracy_score(y_test, predictions_on_genuine_samples_map[uk])
            user_model_recall_map[uk] = recall_score(y_test, predictions_on_genuine_samples_map[uk], average='micro')
            predictions_on_attacks_samples_map[uk] = self._estimator.predict(X_attacks).flatten().tolist()
            user_model_tn_rate_on_attack_samples_map[uk] = accuracy_score(y_attacks, predictions_on_attacks_samples_map[uk])

        results = OneClassResults( 
                user_model_acc_on_genuine_samples_map=user_model_acc_on_genuine_samples_map,
                user_model_recall_map=user_model_recall_map,
                user_model_tn_rate_on_attack_samples_map = user_model_tn_rate_on_attack_samples_map,
                predictions_on_user_samples_map = predictions_on_genuine_samples_map,
                predictions_on_impostor_samples_map = predictions_on_attacks_samples_map,
                hp = one_class_estimators_hp_map)

        return results
