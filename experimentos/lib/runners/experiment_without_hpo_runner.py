from sklearn.base import BaseEstimator

from lib.datasets.dataset import *
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner


class ExperimentWithoutHPORunner(ExperimentRunner):
    _estimator: BaseEstimator

    def __init__(self, dataset: Dataset, estimator: BaseEstimator, exp_name: str, results_repo: ResultsRepository,
                 use_impostor_samples: bool):
        super().__init__(dataset=dataset, use_impostor_samples=use_impostor_samples, exp_name=exp_name,
                         results_repo=results_repo)
        self._estimator = estimator
        self.logger = logging.getLogger(__name__)

    def exec(self) -> None:
        start_time = datetime.now()
        self.logger.info(f"Starting {self._exp_name} at {start_time}")

        self._results_repository.add_hp(hp={'default': [self._estimator.get_params()]}, exp_name=self._exp_name,
                                        date=start_time)

        if self._use_impostor_samples:
            for seed in list(seeds_range):
                self._dataset.set_seed(seed)
                pred_df = pd.DataFrame(self._train_and_predict(seed))
                self._results_repository.add_predictions_frame(predictions_frame=pred_df, date=start_time,
                                                               exp_name=self._exp_name, seed=seed)
        else:
            self._results_repository.add_predictions_frame(predictions_frame=pd.DataFrame(self._train_and_predict()),
                                                           date=start_time, exp_name=self._exp_name)
        
        self._log_experiment_completion(start_time)

    def _train_and_predict(self, seed: int = 0) -> list[pd.Series]:
        pred_frames: list[pd.Series] = []
        for uk in self._dataset.user_keys():
            x_training, y_training = self._get_user_training_vectors(uk)
            self._estimator.fit(x_training.drop(columns=self._dataset.get_drop_columns()), y_training)
            pred_frames += self._test_user_model(estimator=self._estimator, uk=uk, seed=seed)
        return pred_frames
