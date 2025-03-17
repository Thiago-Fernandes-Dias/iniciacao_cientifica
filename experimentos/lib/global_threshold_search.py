from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from lib.cmu_dataset import CMUDataset


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
        best_threshold: float = 0
        best_bacc: float = 0
        self._fit_estimators()
        for threshold in self._thresholds:
            bacc_map: dict[str, float] = {}
            for uk in self._dataset.user_keys():
                self._estimators_map[uk] = self._estimators_map[uk].set_params(**{"threshold": threshold})
                x_g_test, y_g_test = self._dataset.user_test_set(uk)
                x_i_test, y_i_test = self._dataset.impostors_test_set(uk)
                g_pred = self._estimators_map[uk].predict(x_g_test)
                i_pred = self._estimators_map[uk].predict(x_i_test)
                g_recall = accuracy_score(y_g_test, g_pred)
                i_recall = accuracy_score(y_i_test, i_pred)
                bacc_map[uk] = (g_recall + i_recall) / 2
            av_bacc = np.average(list(bacc_map.values())).item()
            if av_bacc > best_bacc:
                best_threshold = threshold
                best_bacc = av_bacc
        return best_threshold

    def _fit_estimators(self) -> None:
        for uk in self._dataset.user_keys():
            self._estimators_map[uk] = self._estimator_factory()
            x_g_training, y_g_training = self._dataset.one_class_training_set(uk)
            self._estimators_map[uk].fit(x_g_training, y_g_training)
