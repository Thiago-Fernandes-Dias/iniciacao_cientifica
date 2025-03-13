from typing import Callable

from sklearn.base import BaseEstimator

from lib.cmu_dataset import CMUDataset
from lib.lightweight_alg import LightWeightAlg
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl


class GlobalThresholdTuning:
    _dataset: CMUDataset
    _thresholds: list[float]
    _estimator_factory: Callable[[], BaseEstimator]

    _estimators_map: dict[str, BaseEstimator] = {}

    def __init__(self, dataset: CMUDataset, estimator_factory: Callable[[], BaseEstimator], thresholds: list[float]):
        self._dataset = dataset
        self._estimator_factory = estimator_factory
        self._thresholds = thresholds

    def search(self) -> float:
        self._fit_estimators()
        best_threshold: float = 0
        best_bacc: float = 0
        for threshold in self._thresholds:
            self._estimators_map[uk] = self._estimators_map[uk].set_params(**{"threshold": threshold})
            far_map: dict[str, float] = {}
            frr_map: dict[str, float] = {}
            for uk in self._dataset.user_keys():
                x_g_test, y_g_test = self._dataset.user_test_set(uk)
                x_i_test, y_i_test = self._dataset.impostors_test_set(uk)
                g_pred = self._estimators_map[uk].predict(x_i_test)
                i_pred = self._estimators_map[uk].set_predict_request()
            bacc = (g_recall + i_recall) / 2
            if bacc > best_bacc:
                best_threshold = threshold
                best_bacc = bacc
        return best_threshold

    def _fit_estimators(self) -> None:
        for uk in self._dataset.user_keys():
            self._estimators_map[uk] = self._estimator_factory()
            x_g_training, y_g_training, x_i_training, y_i_training = self._dataset.two_class_training_set(uk)
            self._estimators_map[uk].fit(x_g_training, y_g_training)
