from typing import Callable, Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid

from lib.constants import GENUINE_LABEL, IMPOSTOR_LABEL
from lib.datasets.dataset import Dataset
from lib.runners.experiment_runner import ExperimentRunner
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner
from lib.utils import create_labels, dict_values_average, seeds_range


class GlobalHPTuning:
    _dataset: Dataset
    _parameter_grid: list[dict[str, Any]]
    _estimator_factory: Callable[[], BaseEstimator]

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator],
                 parameter_grid: list[dict[str, Any]]):
        self._dataset = dataset
        self._estimator_factory = estimator_factory
        self._parameter_grid = parameter_grid

    def search(self) -> dict[str, Any]:
        best_param_config: dict[str, Any] = {}
        best_bacc: float = 0
        for seed in list(seeds_range):
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            for param_config in ParameterGrid(self._parameter_grid):
                users_bacc_map: dict[str, float] = {}
                for uk in self._dataset.user_keys():
                    user_baccs: list[float] = []
                    x_genuine, y_genuine, x_impostor, y_impostor = \
                        self._dataset.two_class_training_set(uk)
                    for gss, iss in zip(cv.split(x_genuine, y_genuine),
                                        cv.split(x_impostor, y_impostor)):
                        x_g_train = x_genuine.drop(columns=self._dataset.get_drop_columns()).iloc[gss[0]]
                        x_g_test = x_genuine.drop(columns=self._dataset.get_drop_columns()).iloc[gss[1]]
                        x_i_test = x_impostor.drop(columns=self._dataset.get_drop_columns()).iloc[iss[1]]
                        estimator = self._estimator_factory().set_params(**param_config)
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
                    best_param_config = param_config
        return best_param_config


def runner_with_global_hpo_factory(ds: Dataset, params_grid: list[dict[str, Any]],
                                   estimator_factory: Callable[[], BaseEstimator]) -> ExperimentRunner:
    global_hp_tuning = GlobalHPTuning(
        dataset=ds,
        parameter_grid=params_grid,
        estimator_factory=estimator_factory,
    )
    best_param_config = global_hp_tuning.search()
    return ExperimentWithoutHPORunner(
        dataset=ds,
        estimator=estimator_factory().set_params(**best_param_config),
        use_impostor_samples=False
    )
