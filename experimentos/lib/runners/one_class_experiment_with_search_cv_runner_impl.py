from typing import Callable

import pandas as pd
from sklearn.model_selection._search import BaseSearchCV

from lib.datasets.dataset import Dataset
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner
from lib.user_model_prediction import UserModelPrediction


class OneClassExperimentWithSearchCVRunnerImpl(OneClassExperimentRunner):
    _estimator_factory: Callable[[int], BaseSearchCV]
    _dataset: Dataset

    def __init__(self, dataset: Dataset, estimator_factory: Callable[[int], BaseSearchCV],
                 use_impostor_samples: bool = False):
        super().__init__(dataset, use_impostor_samples)
        self._estimator_factory = estimator_factory

    def _calculate_predictions(self) -> None:
        pred_frames = list[pd.Series]()
        for random_seed in range(0, 30):
            estimator = self._estimator_factory(random_seed)
            for uk in self._dataset.user_keys():
                x_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
                if self._use_impostor_samples:
                    x_training = pd.concat([x_training, self._X_impostor_training[uk]])
                    y_training = y_training + self._y_impostor_training[uk]
                x_training = x_training.drop(columns=self._dataset.get_columns_to_drop())
                estimator.fit(x_training, y_training)
                if self._one_class_estimators_hp_map[uk] is None:
                    self._one_class_estimators_hp_map[uk] = []
                self._one_class_estimators_hp_map[uk].append(estimator.best_params_)
                X_test = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
                y_test = self._y_genuine_test[uk] + self._y_impostors_test[uk]
                for (_, X), y in zip(X_test.iterrows(), y_test):
                    X_filtered = X.drop(labels=self._dataset.get_columns_to_drop())
                    pred = UserModelPrediction(
                        user_id=uk,
                        expected=y,
                        predicted=self._estimator_factory.predict([X_filtered])[0].item(),
                        session=X[self._dataset.session_key_name()],
                        repetition=X[self._dataset.repetition_key_name()],
                    )
                    pred_frame = pd.Series(pred.to_dict())
                    pred_frames.append(pred_frame)
        self._user_model_predictions = pd.DataFrame(pred_frames)

