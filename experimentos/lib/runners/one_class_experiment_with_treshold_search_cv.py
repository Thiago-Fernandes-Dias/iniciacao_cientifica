from sklearn.model_selection._search import BaseSearchCV

from lib.cmu_dataset import CMUDataset
from lib.one_class_threshold_search_cv import OneClassThresholdSearchCV
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner


class OneClassExperimentWithTresholdSearchCV(OneClassExperimentRunner):
    _estimator: OneClassThresholdSearchCV
    _dataset: CMUDataset

    def __init__(self, dataset: CMUDataset, estimator: OneClassThresholdSearchCV):
        # Second parameter is ignored. MB
        super().__init__(dataset, True)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        for uk in self._dataset.user_keys():
            self._estimator.fit(self._X_genuine_training[uk], self._X_impostor_training[uk])
            self._one_class_estimators_hp_map[uk] = self._estimator.get_params()
            self._predictions_on_genuine_samples_map[uk] = \
                self._estimator.predict(self._X_genuine_test[uk]).flatten().tolist()
            self._predictions_on_attacks_samples_map[uk] = \
                self._estimator.predict(self._X_impostors_test[uk]).flatten().tolist()