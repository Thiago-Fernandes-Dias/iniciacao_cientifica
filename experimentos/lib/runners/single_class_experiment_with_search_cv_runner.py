from lib.cmu_dataset import CMUDataset
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner
from sklearn.model_selection._search import BaseSearchCV

class SingleClassExperimentWithSearchCVRunner(OneClassExperimentRunner):
    _estimator: BaseSearchCV
    _dataset: CMUDataset

    def __init__(self, dataset: CMUDataset, estimator: BaseSearchCV):
        super().__init__(dataset)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        for uk in self._dataset.user_keys():
            self._estimator.fit(self._X_training[uk], self._y_training[uk])
            self._one_class_estimators_hp_map[uk] = self._estimator.best_params_
            self._predictions_on_genuine_samples_map[uk] = self._estimator.predict(self._X_test[uk]).flatten().tolist()
            self._predictions_on_attacks_samples_map[uk] = self._estimator.predict(self._X_attacks[uk]).flatten().tolist()
