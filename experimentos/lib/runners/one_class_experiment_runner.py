from abc import abstractmethod
from lib.datasets.dataset import Dataset
from lib.one_class_results import *
from sklearn.metrics import accuracy_score, recall_score

class OneClassExperimentRunner:
    _dataset: Dataset
    _use_impostor_samples: bool

    def __init__(self, dataset: Dataset, use_impostor_samples: bool = False):
        self._dataset = dataset
        self._use_impostor_samples = use_impostor_samples

    _X_genuine_training: dict[str, pd.DataFrame] = {}
    _y_genuine_training: dict[str, list[int]] = {}
    _X_impostor_training: dict[str, pd.DataFrame] = {}
    _y_impostor_training: dict[str, list[int]] = {}
    _X_genuine_test: dict[str, pd.DataFrame] = {}
    _y_genuine_test: dict[str, list[int]] = {}
    _X_impostors_test: dict[str, pd.DataFrame] = {}
    _y_impostors_test: dict[str, list[int]] = {}
    _predictions_on_genuine_samples_map: dict[str, list[int]] = {}
    _one_class_estimators_hp_map: dict[str, dict[str, object]] = {}
    _predictions_on_attacks_samples_map: dict[str, list[int]] = {}
    _frr: list[UserModelMetric] = []
    _far: list[UserModelMetric] = []

    @abstractmethod 
    def _calculate_predictions(self) -> None:
        pass

    def exec(self) -> OneClassResults:
        self._set_vectors_and_true_labels()
        self._calculate_predictions()
        self._calculate_metrics()

        results = OneClassResults( 
                frr= self._frr,
                far= self._far,
                hp = self._one_class_estimators_hp_map,
                date=datetime.now())
        
        return results

    def _set_vectors_and_true_labels(self) -> None:
        for uk in self._dataset.user_keys():
            self._X_genuine_training[uk], self._y_genuine_training[uk], \
                self._X_impostor_training[uk], self._y_impostor_training[uk] = self._dataset.two_class_training_set(uk)
            self._X_genuine_test[uk], self._y_genuine_test[uk] = self._dataset.user_test_set(uk)
            self._X_impostors_test[uk], self._y_impostors_test[uk] = self._dataset.impostors_test_set(uk)
    
    def _calculate_metrics(self):
        for uk in self._dataset.user_keys():
            frr_value = 1.0 - accuracy_score(self._y_genuine_test[uk], self._predictions_on_genuine_samples_map[uk])
            self._frr.append(UserModelMetric(uk, frr_value))
            far_value = 1.0 - accuracy_score(self._y_impostors_test[uk], self._predictions_on_attacks_samples_map[uk])
            self._far.append(UserModelMetric(uk, far_value))
                
