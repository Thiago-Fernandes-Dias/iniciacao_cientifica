from sklearn.base import BaseEstimator
from lib.cmu_dataset import *
from lib.runners.one_class_experiment_runner import OneClassExperimentRunner

class OneClassExperimentRunnerImpl(OneClassExperimentRunner):
    _estimator: BaseEstimator
    _dataset: CMUDataset

    def __init__(self, dataset: CMUDataset, estimator: BaseEstimator, use_impostor_samples: bool = False):
        super().__init__(dataset, use_impostor_samples)
        self._estimator = estimator

    def _calculate_predictions(self) -> None:
        self._one_class_estimators_hp_map['default'] = self._estimator.get_params()
        for uk in self._dataset.user_keys():
            x_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
            if self._use_impostor_samples:
                x_training = pd.concat([x_training, self._X_impostor_training[uk]])
                y_training = y_training + self._y_impostor_training[uk]
            self._estimator.fit(x_training, y_training)
            self._predictions_on_genuine_samples_map[uk] = \
                self._estimator.predict(self._X_genuine_test[uk]).flatten().tolist()
            self._predictions_on_attacks_samples_map[uk] = \
                self._estimator.predict(self._X_impostors_test[uk]).flatten().tolist()
