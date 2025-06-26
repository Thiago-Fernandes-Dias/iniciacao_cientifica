from abc import abstractmethod
from lib.datasets.dataset import Dataset
from lib.one_class_results import *

from lib.user_model_prediction import UserModelPrediction

class OneClassExperimentRunner:
    _dataset: Dataset
    _use_impostor_samples: bool
    _X_genuine_training: dict[str, pd.DataFrame]
    _y_genuine_training: dict[str, list[int]]
    _X_impostor_training: dict[str, pd.DataFrame]
    _y_impostor_training: dict[str, list[int]]
    _X_genuine_test: dict[str, pd.DataFrame]
    _y_genuine_test: dict[str, list[int]]
    _X_impostors_test: dict[str, pd.DataFrame]
    _y_impostors_test: dict[str, list[int]]
    _predictions_on_genuine_samples_map: dict[str, list[int]]
    _one_class_estimators_hp_map: dict[str, dict[str, object]]
    _user_model_predictions: pd.DataFrame

    def __init__(self, dataset: Dataset, use_impostor_samples: bool = False):
        self._dataset = dataset
        self._use_impostor_samples = use_impostor_samples
        self._X_genuine_training = {}
        self._y_genuine_training = {}
        self._X_impostor_training = {}
        self._y_impostor_training = {}
        self._X_genuine_test = {}
        self._y_genuine_test = {}
        self._X_impostors_test = {}
        self._y_impostors_test = {}
        self._one_class_estimators_hp_map = {}
        self._user_model_predictions = pd.DataFrame()

    @abstractmethod 
    def _calculate_predictions(self) -> None:
        pass

    def exec(self) -> OneClassResults:
        self._reset_from_previous_execution()
        self._set_vectors_and_true_labels()
        self._calculate_predictions()
        results = OneClassResults( 
                user_model_predictions=self._user_model_predictions,
                hp = pd.DataFrame.from_dict(self._one_class_estimators_hp_map, orient='index'),
                date=datetime.now())
        return results

    def _set_vectors_and_true_labels(self) -> None:
        for uk in self._dataset.user_keys():
            self._X_genuine_training[uk], self._y_genuine_training[uk], \
                self._X_impostor_training[uk], self._y_impostor_training[uk] = self._dataset.two_class_training_set(uk)
            self._X_genuine_test[uk], self._y_genuine_test[uk] = self._dataset.user_test_set(uk)
            self._X_impostors_test[uk], self._y_impostors_test[uk] = self._dataset.impostors_test_set(uk)
    
    def _reset_from_previous_execution(self):
        self._X_genuine_training.clear()
        self._y_genuine_training.clear()
        self._X_impostor_training.clear()
        self._y_impostor_training.clear()
        self._X_genuine_test.clear()
        self._y_genuine_test.clear()
        self._X_impostors_test.clear()
        self._y_impostors_test.clear()
        self._user_model_predictions.drop(self._user_model_predictions.index, inplace=True)
        self._one_class_estimators_hp_map.clear()
