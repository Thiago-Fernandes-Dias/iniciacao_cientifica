from sklearn.base import BaseEstimator
from cmu_dataset import *
from runners.one_class_experiment_runner import OneClassExperimentRunner

class SingleClassExperimentRunner(OneClassExperimentRunner):
    _estimator: BaseEstimator
    _dataset: CMUDataset

    def __init__(self, dataset: CMUDataset, estimator: BaseEstimator):
        super().__init__(dataset)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        self._one_class_estimators_hp_map['default'] = self._estimator.get_params()
        for uk in self._dataset.user_keys():
            self._estimator.fit(self._X_training[uk], self._y_training[uk])
            self._predictions_on_genuine_samples_map[uk] = self._estimator.predict(self._X_test[uk]).flatten().tolist()
            self._predictions_on_attacks_samples_map[uk] = self._estimator.predict(self._X_attacks[uk]).flatten().tolist()
