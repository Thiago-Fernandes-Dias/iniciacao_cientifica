from cmu_dataset import CMUDataset
from sklearn.model_selection._search import BaseSearchCV
from runners.two_class_experiment_runner import TwoClassExperimentRunner

class OneVsRestExperimentWithSearchCVRunner(TwoClassExperimentRunner):
    _cmu_database: CMUDataset
    _estimator: BaseSearchCV
    
    def __init__(self, cmu_database: CMUDataset, estimator: BaseSearchCV):
        super().__init__(cmu_database)
        self._estimator = estimator

    def _calculate_predictions(self):
        for uk in self._cmu_database.user_keys():
            self._estimator.fit(self._X_training[uk], self._y_training[uk])
            self._estimators_hp_map[uk] = self._estimator.best_params_
            self._predictions_on_genuine_samples_map[uk] = self._estimator.predict(self._X_user_test[uk]).flatten().tolist()
            self._predictions_on_impostor_samples_map[uk] = self._estimator.predict(self._X_other_test[uk]).flatten().tolist()