from typing import Callable, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid

from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL
from lib.datasets.dataset import Dataset
from lib.utils import create_labels, dict_values_average


class GlobalHPTuning:
    _dataset: Dataset
    _parameters_grid: ParameterGrid
    _estimator_factory: Callable[[], BaseEstimator]
    _cv: KFold

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator], params_grid: ParameterGrid,
                 cv: KFold):
        self._dataset = dataset
        self._estimator_factory = estimator_factory
        self._parameters_grid = params_grid
        self._cv = cv

    def search(self) -> dict[str, Any]:
        best_params_config: dict[str, Any] = {}
        best_bacc: float = 0
        for params_setting in self._parameters_grid:
            users_bacc_map: dict[str, float] = {}
            for uk in self._dataset.user_keys():
                user_baccs: list[float] = []
                x_genuine, y_genuine, x_impostor, y_impostor = \
                    self._dataset.two_class_training_set(uk)
                for gss, iss in zip(self._cv.split(x_genuine, y_genuine),
                                    self._cv.split(x_impostor, y_impostor)):
                    x_g_train = x_genuine.drop(columns=self._dataset.get_columns_to_drop()).iloc[gss[0]]
                    x_g_test = x_genuine.drop(columns=self._dataset.get_columns_to_drop()).iloc[gss[1]]
                    x_i_test = x_impostor.drop(columns=self._dataset.get_columns_to_drop()).iloc[iss[1]]
                    estimator = self._estimator_factory().set_params(**params_setting)
                    y_g_train = create_labels(x_g_train, GENUINE_LABEL)
                    estimator.fit(x_g_train, y_g_train)
                    g_pred = estimator.predict(x_g_test)
                    i_pred = estimator.predict(x_i_test)
                    g_recall = accuracy_score(create_labels(x_g_test, GENUINE_LABEL), g_pred)
                    i_recall = accuracy_score(create_labels(x_i_test, IMPOSTOR_LABEL), i_pred)
                    user_baccs.append((g_recall + i_recall) / 2)
                users_bacc_map[uk] = np.average(user_baccs).item()
            average_bacc = dict_values_average(users_bacc_map)
            if average_bacc > best_bacc:
                best_bacc = average_bacc
                best_params_config = params_setting
        return best_params_config
