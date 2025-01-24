from abc import abstractmethod
from cmu_dataset import *
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score, recall_score
from two_class_results import TwoClassResults

class TwoClassExperimentRunner:
    _cmu_database: CMUDataset

    _X_training: dict[str, pd.DataFrame] = {}
    _y_training: dict[str, list[int]] = {}
    _X_user_test: dict[str, pd.DataFrame] = {}
    _y_user_test: dict[str, list[int]] = {}
    _X_other_test: dict[str, pd.DataFrame] = {}
    _y_other_test: dict[str, list[int]] = {}
    _estimators_hp_map: dict[str, dict[str, object]] = {}
    _two_class_bacc_map: dict[str, float] = {}
    _two_class_recall_map: dict[str, float] = {}
    _predictions_on_genuine_samples_map: dict[str, list[int]] = {}
    _predictions_on_impostor_samples_map: dict[str, list[int]] = {}

    def __init__(self, dataset: CMUDataset) -> None:
        self._cmu_database = dataset
    
    @abstractmethod
    def _calculate_predictions(self) -> None:
        pass
    
    def _set_vectors_and_true_labels(self) -> None:
        for uk in self._cmu_database.user_keys():
            self._X_training[uk], self._y_training[uk] = \
                self._cmu_database.one_vs_rest_training_rows(uk)
            self._X_user_test[uk], self._y_user_test[uk], self._X_other_test[uk], self._y_other_test[uk] = \
                self._cmu_database.one_vs_rest_test_rows(uk)
    
    def _calculate_metrics(self) -> None:
        for uk in self._cmu_database.user_keys():
            predictions = self._predictions_on_genuine_samples_map[uk] + self._predictions_on_impostor_samples_map[uk]
            y_test = self._y_user_test[uk] + self._y_other_test[uk]
            self._two_class_bacc_map[uk] = balanced_accuracy_score(y_test, predictions)
            self._two_class_recall_map[uk] = recall_score(y_test, predictions, average='micro')
    
    def exec(self) -> TwoClassResults:
        self._set_vectors_and_true_labels()
        self._calculate_predictions()
        self._calculate_metrics()

        results = TwoClassResults(two_class_bacc_map=self._two_class_bacc_map, 
                                  two_class_recall_map=self._two_class_recall_map, 
                                  predictions_on_user_samples_map=self._predictions_on_genuine_samples_map,
                                  predictions_on_impostor_samples_map=self._predictions_on_impostor_samples_map,
                                  hp=self._estimators_hp_map)

        return results
