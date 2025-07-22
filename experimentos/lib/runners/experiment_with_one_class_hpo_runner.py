from datetime import datetime
from typing import Callable, Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold

from lib.datasets.dataset import Dataset
from lib.estimators.one_class_search_cv import OneClassSearchCV
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner
from lib.utils import seeds_range


class ExperimentWithOneClassHPORunner(ExperimentRunner):
    _estimator_factory: Callable[[], BaseEstimator]
    _params_grid: list[dict[str, Any]]
    _dataset: Dataset

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[], BaseEstimator],
                 params_grid: list[dict[str, Any]], exp_name: str, results_repo: ResultsRepository) -> None:
        super().__init__(dataset=dataset, exp_name=exp_name, use_impostor_samples=True, results_repo=results_repo)
        self._estimator_factory = estimator_factory
        self._params_grid = params_grid

    def exec(self) -> None:
        date = datetime.now()
        
        for seed in list(seeds_range):
            pred_series = list[pd.Series]()

            self._dataset.set_seed(seed)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            one_class_search_cv = OneClassSearchCV(estimator=self._estimator_factory(), cv=cv,
                                                   params_grid=self._params_grid)

            for uk in self._dataset.user_keys():
                x_training, y_training = self._get_user_training_vectors(uk)
                one_class_search_cv.fit(x_training, y_training, user=uk, user_label=self._dataset.user_key_name(),
                                        drop_columns=self._dataset.get_drop_columns())

                if not uk in self._one_class_estimators_hp_map:
                    self._one_class_estimators_hp_map[uk] = []
                self._one_class_estimators_hp_map[uk].append(one_class_search_cv.get_params())

                pred_series += self._test_user_model(estimator=one_class_search_cv, uk=uk, seed=seed)
            
            pred_frame = pd.DataFrame(pred_series)
            self._results_repository.add_predictions_frame(predictions_frame=pred_frame, 
                                                           date=date, seed=seed, exp_name=self._exp_name)
        
        self._results_repository.add_hp(hp=self._one_class_estimators_hp_map, exp_name=self._exp_name, date=date)


