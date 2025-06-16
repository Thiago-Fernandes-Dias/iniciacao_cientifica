from lib.datasets.dataset import Dataset
from lib.one_class_threshold_search_cv import OneClassThresholdSearchCV
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner
from lib.user_model_prediction import UserModelPrediction

class OneClassExperimentWithThresholdSearchCV(OneClassExperimentRunner):
    _estimator: OneClassThresholdSearchCV
    _dataset: Dataset

    def __init__(self, dataset: Dataset, estimator: OneClassThresholdSearchCV):
        # Second parameter is ignored. MB
        super().__init__(dataset, True)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        for uk in self._dataset.user_keys():
            self._estimator.fit(self._X_genuine_training[uk], self._X_impostor_training[uk])
            self._one_class_estimators_hp_map[uk] = self._estimator.get_params()
            for x_test, y_test in zip(self._X_genuine_test[uk], self._y_genuine_test[uk]):
                pred = self._estimator.predict([x_test])[0]
                user_model_prediction = UserModelPrediction(
                    user_key=uk,
                    expected=y_test,
                    predicted=pred,
                    session=x_test[self._dataset._session_key_name(uk)],
                    repetition=x_test[self._dataset._repetition_key_name(uk)],
                )
                self._user_model_predictions.append(user_model_prediction)