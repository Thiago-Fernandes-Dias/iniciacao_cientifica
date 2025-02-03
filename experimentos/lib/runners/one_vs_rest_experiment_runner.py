from sklearn.base import BaseEstimator
from lib.cmu_dataset import CMUDataset
from lib.runners.two_class_experiment_runner import TwoClassExperimentRunner

class OneVsRestExperimentRunner(TwoClassExperimentRunner):
    _cmu_database: CMUDataset
    _estimator: BaseEstimator
    
    def __init__(self, cmu_database: CMUDataset, estimator: BaseEstimator):
        super().__init__(cmu_database)
        self._estimator = estimator

    def _calculate_predictions(self):
        self._estimators_hp_map['default'] = self._estimator.get_params()
        for uk in self._cmu_database.user_keys():
            self._estimator.fit(self._X_training[uk], self._y_training[uk])
            self._predictions_on_genuine_samples_map[uk] = self._estimator.predict(self._X_user_test[uk]).flatten().tolist()
            self._predictions_on_impostor_samples_map[uk] = self._estimator.predict(self._X_other_test[uk]).flatten().tolist()