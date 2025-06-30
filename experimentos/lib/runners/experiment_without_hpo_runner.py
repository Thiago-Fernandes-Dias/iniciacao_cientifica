import numpy as np

from sklearn.base import BaseEstimator
from lib.datasets.dataset import *
from lib.runners.experiment_runner import ExperimentRunner
from lib.user_model_prediction import UserModelPrediction

class ExperimentWithoutHPORunner(ExperimentRunner):
    _estimator: BaseEstimator

    def __init__(self, dataset: Dataset, estimator: BaseEstimator, use_impostor_samples: bool = False):
        super().__init__(dataset, use_impostor_samples)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        self._one_class_estimators_hp_map['default'] = [self._estimator.get_params()]
        pred_frames = list[pd.Series]()
        for uk in self._dataset.user_keys():
            x_training, y_training = self._get_user_training_vectors(uk)
            self._estimator.fit(x_training, y_training)
            pred_frames += self._calculate_user_model_predictions(estimator=self._estimator, uk=uk, seed=None)
        self._user_model_predictions = pd.DataFrame(pred_frames)
