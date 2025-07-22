from datetime import datetime
import numpy as np

from sklearn.base import BaseEstimator
from lib.datasets.dataset import *
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner

class ExperimentWithoutHPORunner(ExperimentRunner):
    _estimator: BaseEstimator

    def __init__(self, dataset: Dataset, estimator: BaseEstimator, exp_name: str, results_repo: ResultsRepository, 
                 use_impostor_samples: bool = False):
        super().__init__(dataset=dataset, use_impostor_samples=use_impostor_samples, exp_name=exp_name,
                         results_repo=results_repo)
        self._estimator = estimator

    def exec(self) -> None:
        date = datetime.now()

        self._results_repository.add_hp(hp={'default': [self._estimator.get_params()]}, exp_name=self._exp_name,
                                        date=date)

        if self._use_impostor_samples:
            pred_frames = list[pd.Series]()

            for seed in list(seeds_range):
                self._dataset.set_seed(seed)
                pred_frames += self._train_and_predict(seed)
            
            self._results_repository.add_predictions_frame(predictions_frame=pd.DataFrame(pred_frames), date=date,
                                                           exp_name=self._exp_name, seed=seed)
        else:
            self._results_repository.add_predictions_frame(predictions_frame=pd.DataFrame(self._train_and_predict()),
                                                           date=date, exp_name=self._exp_name, seed=0)    

    def _train_and_predict(self, seed: int | None = None):
        pred_frames = list[pd.Series]()
        for uk in self._dataset.user_keys():
            x_training, y_training = self._get_user_training_vectors(uk)
            self._estimator.fit(x_training.drop(columns=self._dataset.get_drop_columns()), y_training)
            pred_frames += self._test_user_model(estimator=self._estimator, uk=uk, seed=seed)
        return pred_frames
