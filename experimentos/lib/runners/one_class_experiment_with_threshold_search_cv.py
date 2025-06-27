import pandas as pd
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
            X_g_training = self._X_genuine_training[uk].drop(columns=self._dataset.get_columns_to_drop())
            X_i_training = self._X_impostor_training[uk].drop(columns=self._dataset.get_columns_to_drop())
            self._estimator.fit(X_g_training, X_i_training)
            self._one_class_estimators_hp_map[uk] = self._estimator.get_params()
            X_g_test = self._X_genuine_test[uk]
            y_g_test = self._y_genuine_test[uk]
            pred_frames = list[pd.Series]()
            for (_, X), y in zip(X_g_test.iterrows(), y_g_test):
                X_filtered = X.drop(labels=self._dataset.get_columns_to_drop())
                pred = UserModelPrediction(
                    user_id=uk,
                    expected=y,
                    predicted=self._estimator.predict([X_filtered])[0].item(),
                    session=X[self._dataset.session_key_name(uk)],
                    repetition=X[self._dataset.repetition_key_name(uk)],
                )
                pred_frame = pd.Series(pred.to_dict())
                pred_frames.append(pred_frame)
            self._user_model_predictions = pd.DataFrame(pred_frames)