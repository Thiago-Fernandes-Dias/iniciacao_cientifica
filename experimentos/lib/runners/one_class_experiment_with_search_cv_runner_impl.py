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
            training_vectors, training_labels = self._X_genuine_training[uk], self._y_genuine_training[uk]
            if self._use_impostor_samples:
                training_vectors = pd.concat([training_vectors, self._X_impostor_training[uk]])
                training_labels = training_labels + self._y_impostor_training[uk]
            self._estimator.fit(training_vectors, training_labels)
            self._one_class_estimators_hp_map[uk] = self._estimator.best_params_
            test_vectors = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
            test_labels = self._y_genuine_test[uk] + self._y_impostors_test[uk]
            for x_test, y_test in zip(test_vectors, test_labels):
                pred = UserModelPrediction(
                    user_key=uk,
                    expected=y_test,
                    predicted=self._estimator.predict([x_test])[0],
                    session=x_test[self._dataset._session_key_name(uk)],
                    repetition=x_test[self._dataset._repetition_key_name(uk)],
                )
                self._user_model_predictions.append(pred)
