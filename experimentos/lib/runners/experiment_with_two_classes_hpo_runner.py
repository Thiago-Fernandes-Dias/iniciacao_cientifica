from datetime import datetime
from typing import Callable, Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from lib.datasets.dataset import Dataset
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner
from lib.utils import seeds_range


class ExperimentWithTwoClassesRunner(ExperimentRunner):
    _estimator_factory: Callable[[], BaseEstimator]
    _dataset: Dataset
    _param_grid: list[dict[str, Any]]

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator],
                 params_grid: list[dict[str, Any]], exp_name: str, results_repo: ResultsRepository) -> None:
        super().__init__(dataset=dataset, use_impostor_samples=True, exp_name=exp_name, results_repo=results_repo)
        self._estimator_factory = estimator_factory
        self._param_grid = params_grid

    def exec(self) -> None:
        date = datetime.now()

        for seed in list(seeds_range):
            pred_series = list[pd.Series]()

            self._dataset.set_seed(seed)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            estimator = GridSearchCV(estimator=self._estimator_factory(), cv=cv, n_jobs=-1, param_grid=self._param_grid,
                                     scoring="accuracy")

            for uk in self._dataset.user_keys():
                x_training, y_training = self._get_user_training_vectors(uk)
                estimator.fit(x_training.drop(columns=self._dataset.get_drop_columns()), y_training)

                if not uk in self._one_class_estimators_hp_map:
                    self._one_class_estimators_hp_map[uk] = []
                self._one_class_estimators_hp_map[uk].append(estimator.best_params_)

                pred_series += self._test_user_model(estimator, uk, seed)
            
            pred_frame = pd.DataFrame(pred_series)
            self._results_repository.add_predictions_frame(predictions_frame=pred_frame, seed=seed,
                                                           exp_name=self._exp_name, date=date)

        self._results_repository.add_hp(hp=self._one_class_estimators_hp_map, exp_name=self._exp_name, date=date)