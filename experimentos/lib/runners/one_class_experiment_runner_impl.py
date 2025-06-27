import numpy as np

from sklearn.base import BaseEstimator
from lib.datasets.dataset import *
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner
from lib.user_model_prediction import UserModelPrediction

class OneClassExperimentRunnerImpl(OneClassExperimentRunner):
    _estimator: BaseEstimator

    def __init__(self, dataset: Dataset, estimator: BaseEstimator, use_impostor_samples: bool = False):
        super().__init__(dataset, use_impostor_samples)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        self._one_class_estimators_hp_map['default'] = self._estimator.get_params()
        for uk in self._dataset.user_keys():
            X_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
            if self._use_impostor_samples:
                X_training = pd.concat([X_training, self._X_impostor_training[uk]])
                y_training = y_training + self._y_impostor_training[uk]
            X_training_filtered = X_training.drop(columns=self._dataset.get_columns_to_drop())
            self._estimator.fit(X_training_filtered, y_training)
            X_test = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
            y_test = self._y_genuine_test[uk] + self._y_impostors_test[uk]
            pred_frames = list[pd.Series]()
            for (_, X), y in zip(X_test.iterrows(), y_test):
                X_filtered = X.drop(labels=self._dataset.get_columns_to_drop())
                pred = UserModelPrediction(
                    user_id=uk,
                    expected=y,
                    predicted=self._estimator.predict(pd.DataFrame([X_filtered]))[0].item(),
                    session=X[self._dataset._session_key_name()],
                    repetition=X[self._dataset._repetition_key_name()],
                )
                pred_frame = pd.Series(pred.to_dict())
                pred_frames.append(pred_frame)
            self._user_model_predictions = pd.DataFrame(pred_frames)
