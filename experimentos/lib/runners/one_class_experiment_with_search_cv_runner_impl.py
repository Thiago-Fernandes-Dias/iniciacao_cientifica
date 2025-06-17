import pandas as pd

from lib.datasets.dataset import Dataset
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner
from lib.user_model_prediction import UserModelPrediction
from sklearn.model_selection._search import BaseSearchCV

class OneClassExperimentWithSearchCVRunnerImpl(OneClassExperimentRunner):
    _estimator: BaseSearchCV
    _dataset: Dataset

    def __init__(self, dataset: Dataset, estimator: BaseSearchCV, use_impostor_samples: bool = False):
        super().__init__(dataset, use_impostor_samples)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        for uk in self._dataset.user_keys():
            x_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
            if self._use_impostor_samples:
                x_training = pd.concat([x_training, self._X_impostor_training[uk]])
                y_training = y_training + self._y_impostor_training[uk]
            x_training = x_training.drop(columns=self._dataset._drop_columns())
            self._estimator.fit(x_training, y_training)
            self._one_class_estimators_hp_map[uk] = self._estimator.best_params_
            X_test = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
            y_test = self._y_genuine_test[uk] + self._y_impostors_test[uk]
            for (_, X), y in zip(X_test.iterrows(), y_test):
                X_filtered = X.drop(columns=self._dataset._drop_columns())
                pred = UserModelPrediction(
                    user_key=uk,
                    expected=y,
                    predicted=self._estimator.predict([X_filtered])[0],
                    session=X[self._dataset._session_key_name(uk)],
                    repetition=X[self._dataset._repetition_key_name(uk)],
                )
                self._user_model_predictions.append(pred)
