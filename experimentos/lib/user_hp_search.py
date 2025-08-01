import logging
from datetime import datetime
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid

from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL
from lib.datasets.dataset import Dataset
from lib.utils import create_labels, dict_values_average


class UserHPTuning:
    _dataset: Dataset
    _parameter_grid: list[dict[str, Any]]
    _estimator_factory: Callable[[], BaseEstimator]
    _use_impostor_samples: bool
    _seed: int

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator],
                 parameter_grid: list[dict[str, Any]], use_impostor_samples: bool, seed: int):
        self._dataset = dataset
        self._estimator_factory = estimator_factory
        self._parameter_grid = parameter_grid
        self._use_impostor_samples = use_impostor_samples
        self._seed = seed
        self.logger = logging.getLogger(__name__)

    def search(self) -> dict[str, Any]:
        self.logger.info(f"Starting user hpo search with seed {self._seed}")
        start_time = datetime.now()

        user_best_param_config_map: dict[str, Any] = {}
        cv = KFold(n_splits=5, shuffle=True, random_state=self._seed)

        for uk in self._dataset.user_keys():
            best_bacc = 0.0
            for param_config in ParameterGrid(self._parameter_grid):
                split_baccs = []
                x_genuine, y_genuine, x_impostor, y_impostor = \
                    self._dataset.two_class_training_set(uk)
                for gss, iss in zip(cv.split(x_genuine, y_genuine),
                                    cv.split(x_impostor, y_impostor)):
                    x_train = x_genuine.drop(columns=self._dataset.get_drop_columns()).iloc[gss[0]]
                    x_g_test = x_genuine.drop(columns=self._dataset.get_drop_columns()).iloc[gss[1]]
                    x_i_test = x_impostor.drop(columns=self._dataset.get_drop_columns()).iloc[iss[1]]
                    estimator = self._estimator_factory().set_params(**param_config)
                    y_train = create_labels(x_train, GENUINE_LABEL)
                    if self._use_impostor_samples:
                        x_i_train = x_impostor.drop(columns=self._dataset.get_drop_columns()).iloc[gss[1]]
                        x_train = pd.concat([x_train, x_i_train])
                        y_train = y_train + create_labels(x_i_train, IMPOSTOR_LABEL)
                    estimator.fit(x_train, y_train)
                    g_pred = estimator.predict(x_g_test)
                    i_pred = estimator.predict(x_i_test)
                    g_recall = accuracy_score(create_labels(x_g_test, GENUINE_LABEL), g_pred)
                    i_recall = accuracy_score(create_labels(x_i_test, IMPOSTOR_LABEL), i_pred)
                    split_baccs.append((g_recall + i_recall) / 2)
                average_bacc = np.average(split_baccs).item()
                if average_bacc > best_bacc:
                    best_bacc = average_bacc
                    user_best_param_config_map[uk] = param_config

        minutes_elapsed = (datetime.now() - start_time) // 60
        self.logger.info(f"User hpo search with seed {self._seed} finished. Time elapsed: {minutes_elapsed} minutes")

        return user_best_param_config_map
