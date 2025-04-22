from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from lib.dataset import Dataset
from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL
from lib.utils import create_labels, dict_values_average


class GlobalThresholdTuning:
    _dataset: Dataset
    _thresholds: list[float]
    _estimator_factory: Callable[[], BaseEstimator]
    _cv: KFold

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator], thresholds: list[float], cv: KFold):
        self._dataset = dataset
        self._estimator_factory = estimator_factory
        self._thresholds = thresholds
        self._cv = cv

    def search(self) -> float:
        best_threshold: float = 0
        best_bacc: float = 0
        for threshold in self._thresholds:
            users_bacc_map: dict[str, float] = {}
            for uk in self._dataset.user_keys():
                user_baccs: list[float] = []
                x_genuine, y_genuine, x_impostor, y_impostor = \
                    self._dataset.two_class_training_set(uk)
                for gss, iss in zip(self._cv.split(x_genuine, y_genuine),
                                    self._cv.split(x_impostor, y_impostor)):
                    x_g_train = x_genuine.iloc[gss[0]]
                    x_g_test = x_genuine.iloc[gss[1]]
                    x_i_test = x_impostor.iloc[iss[1]]
                    estimator = self._estimator_factory().set_params(**{'threshold': threshold})
                    y_g_train = create_labels(x_g_train, GENUINE_LABEL)
                    estimator.fit(x_g_train, y_g_train)
                    g_pred = estimator.predict(x_g_test)
                    i_pred =  estimator.predict(x_i_test)
                    g_recall = accuracy_score(create_labels(x_g_test, GENUINE_LABEL), g_pred)
                    i_recall = accuracy_score(create_labels(x_i_test, IMPOSTOR_LABEL), i_pred)
                    user_baccs.append((g_recall + i_recall) / 2)
                users_bacc_map[uk] = np.average(user_baccs).item()
            average_bacc = dict_values_average(users_bacc_map)
            if (average_bacc > best_bacc):
                best_bacc = average_bacc
                best_threshold = threshold
        return best_threshold
