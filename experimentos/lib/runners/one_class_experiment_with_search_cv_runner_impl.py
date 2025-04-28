from lib.datasets.dataset import Dataset
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner
from sklearn.model_selection._search import BaseSearchCV

class OneClassExperimentWithSearchCVRunnerImpl(OneClassExperimentRunner):
    _estimator: BaseSearchCV
    _dataset: Dataset

    def __init__(self, dataset: Dataset, estimator: BaseSearchCV, use_impostor_samples: bool = False):
        super().__init__(dataset, use_impostor_samples)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        for uk in self._dataset.user_keys():
            self._estimator.fit(self._X_genuine_training[uk], self._y_genuine_training[uk])
            self._one_class_estimators_hp_map[uk] = self._estimator.best_params_
            self._predictions_on_genuine_samples_map[uk] = \
                self._estimator.predict(self._X_genuine_test[uk]).flatten().tolist()
            self._predictions_on_attacks_samples_map[uk] = \
                self._estimator.predict(self._X_impostors_test[uk]).flatten().tolist()
