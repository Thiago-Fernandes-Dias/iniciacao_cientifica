from typing import Self, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, BaseCrossValidator

from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL
from lib.utils import create_labels


class OneClassSearchCV:
    _estimator: BaseEstimator
    _params_grid: list[dict[str, Any]]
    _cv: BaseCrossValidator
    _best_params_config: dict[str, Any]

    def __init__(self, estimator: BaseEstimator, params_grid: list[dict[str, Any]], cv: BaseCrossValidator):
        self._estimator = estimator
        self._params_grid = params_grid
        self._cv = cv

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._best_params_config

    def set_params(self, **params) -> Self:
        self._estimator = self._estimator.set_params(**params)

    def fit(self, x: pd.DataFrame, y: list[int], user: str, user_label: str, drop_columns: list[str]) -> None:
        self._best_params_config = {}
        best_bacc = 0.0
        for params_config in ParameterGrid(self._params_grid):
            cv_bacc: list[float] = []
            for split in self._cv.split(x, y):
                x_training, x_test = x.iloc[split[0]], x.iloc[split[1]]
                x_g_training = x_training[x_training[user_label] == user].drop(columns=drop_columns)
                x_g_test, x_i_test = \
                    x_test[x_test[user_label] == user].drop(columns=drop_columns), x_test[x_test[user_label] != user].drop(columns=drop_columns)
                self._estimator = self._estimator.set_params(**params_config)
                self._estimator.fit(x_g_training, create_labels(x_g_training, GENUINE_LABEL))
                g_pred = self._estimator.predict(x_g_test)
                i_pred = self._estimator.predict(x_i_test)
                g_recall = accuracy_score(create_labels(x_g_test, GENUINE_LABEL), g_pred)
                i_recall = accuracy_score(create_labels(x_i_test, IMPOSTOR_LABEL), i_pred)
                cv_bacc.append((g_recall + i_recall) / 2)
            average_bacc = np.average(cv_bacc).item()
            if average_bacc > best_bacc:
                best_bacc = average_bacc
                self._best_params_config = params_config
        self._estimator = self._estimator.set_params(**self._best_params_config)
        x_genuine = x[x[user_label] == user].drop(columns=drop_columns)
        y_genuine = create_labels(x_genuine, GENUINE_LABEL)
        self._estimator.fit(x_genuine, y_genuine)

    def predict(self, x: pd.DataFrame):
        return self._estimator.predict(x)
