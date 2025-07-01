from typing import Callable, Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from lib.datasets.dataset import Dataset
from lib.estimators.one_class_search_cv import OneClassSearchCV
from lib.runners.experiment_runner import ExperimentRunner
from lib.utils import seeds_range


class ExperimentWithOneClassHPORunner(ExperimentRunner):
    _estimator_factory: Callable[[], BaseEstimator]
    _params_grid: list[dict[str, Any]]
    _dataset: Dataset

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator],
                 params_grid: list[dict[str, Any]]) -> None:
        super().__init__(dataset, use_impostor_samples=False)
        self._estimator_factory = estimator_factory
        self._params_grid = params_grid

    def _calculate_predictions(self) -> None:
        pred_frames = list[pd.Series]()
        for seed in list(seeds_range):
            self._dataset.set_seed(seed)
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
            one_class_search_cv = OneClassSearchCV(estimator=self._estimator_factory(), cv=cv,
                                                   params_grid=self._params_grid)
            for uk in self._dataset.user_keys():
                x_g_training = self._X_genuine_training[uk].drop(columns=self._dataset.get_drop_columns())
                x_i_training = self._X_impostor_training[uk].drop(columns=self._dataset.get_drop_columns())
                one_class_search_cv.fit(x_g_training, x_i_training)
                if not uk in self._one_class_estimators_hp_map:
                    self._one_class_estimators_hp_map[uk] = []
                self._one_class_estimators_hp_map[uk].append(one_class_search_cv.get_params())
                pred_frames += self._test_user_model(estimator=one_class_search_cv, uk=uk, seed=seed)
        self._user_model_predictions = pd.DataFrame(pred_frames)
