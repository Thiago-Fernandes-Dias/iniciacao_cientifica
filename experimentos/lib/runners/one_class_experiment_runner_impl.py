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
            x_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
            if self._use_impostor_samples:
                x_training = pd.concat([x_training, self._X_impostor_training[uk]])
                y_training = y_training + self._y_impostor_training[uk]
            x_training = x_training.drop(columns=self._dataset._drop_columns())
            self._estimator.fit(x_training, y_training)
            test_vectors = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
            test_labels = self._y_genuine_test[uk] + self._y_impostors_test[uk]
            for x_test, y_test in zip(test_vectors.iterrows(), test_labels):
                pred = UserModelPrediction(
                    user_key=uk,
                    expected=y_test,
                    predicted=self._estimator.predict(pd.DataFrame([x_test[1]]))[0],
                    session=x_test[1][self._dataset._session_key_name()],
                    repetition=x_test[1][self._dataset._repetition_key_name()],
                )
                self._user_model_predictions.append(pred)
