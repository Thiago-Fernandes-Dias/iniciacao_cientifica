from abc import abstractmethod
from lib.cmu_dataset import CMUDataset
from lib.one_class_results import *
from sklearn.metrics import accuracy_score, recall_score

class OneClassExperimentRunner:
    _dataset: CMUDataset
    _use_impostor_samples: bool

    def __init__(self, dataset: CMUDataset, use_impostor_samples: bool = False):
        self._dataset = dataset
        self._use_impostor_samples = use_impostor_samples

    _X_training: dict[str, pd.DataFrame] = {}
    _X_genuine_test: dict[str, pd.DataFrame] = {}
    _y_training: dict[str, list[int]] = {}
    _y_genuine_test: dict[str, list[int]] = {}
    _X_impostors_test: dict[str, pd.DataFrame] = {}
    _y_impostors_test: dict[str, list[int]] = {}
    _predictions_on_genuine_samples_map: dict[str, list[int]] = {}
    _one_class_estimators_hp_map: dict[str, dict[str, object]] = {}
    _predictions_on_attacks_samples_map: dict[str, list[ int ]] = {}
    _tp_rate_map: dict[str, float] = {}
    _user_model_recall_map: dict[str, float] = {}
    _tn_rate_map: dict[str, float] = {}

    @abstractmethod 
    def _calculate_predictions(self) -> None:
        pass

    def exec(self) -> OneClassResults:
        self._set_vectors_and_true_labels()
        self._calculate_predictions()
        self._calculate_metrics()

        results = OneClassResults( 
                tp_rate_map = self._tp_rate_map,
                tn_rate_map= self._tn_rate_map,
                predictions_on_user_samples_map = self._predictions_on_genuine_samples_map,
                predictions_on_impostor_samples_map = self._predictions_on_attacks_samples_map,
                hp = self._one_class_estimators_hp_map)
        
        return results

    def _set_vectors_and_true_labels(self) -> None:
        for uk in self._dataset.user_keys():
            if self._use_impostor_samples:
                self._X_training[uk], self._y_training[uk] = self._dataset.two_class_training_set(uk)
            else:
                self._X_training[uk], self._y_training[uk] = self._dataset.one_class_training_set(uk)
            self._X_genuine_test[uk], self._y_genuine_test[uk] = self._dataset.user_test_set(uk)
            self._X_impostors_test[uk], self._y_impostors_test[uk] = self._dataset.impostors_test_set(uk)
    
    def _calculate_metrics(self):
        for uk in self._dataset.user_keys():
            self._tp_rate_map[uk] = \
                accuracy_score(self._y_genuine_test[uk], self._predictions_on_genuine_samples_map[uk])
            self._tn_rate_map[uk] = \
                accuracy_score(self._y_impostors_test[uk], self._predictions_on_attacks_samples_map[uk])
