from abc import abstractmethod
from cmu_dataset import CMUDataset
from one_class_results import *
from sklearn.metrics import accuracy_score, recall_score

class OneClassExperimentRunner:
    _dataset: CMUDataset

    def __init__(self, dataset: CMUDataset):
        self._dataset = dataset

    _X_training: dict[str, pd.DataFrame] = {}
    _X_test: dict[str, pd.DataFrame] = {}
    _y_training: dict[str, list[int]] = {}
    _y_test: dict[str, list[int]] = {}
    _X_attacks: dict[str, pd.DataFrame] = {}
    _y_attacks: dict[str, list[int]] = {}
    _predictions_on_genuine_samples_map: dict[str, list[int]] = {}
    _one_class_estimators_hp_map: dict[str, dict[str, object]] = {}
    _predictions_on_attacks_samples_map: dict[str, list[ int ]] = {}
    _user_model_acc_on_genuine_samples_map: dict[str, float] = {}
    _user_model_recall_map: dict[str, float] = {}
    _user_model_tn_rate_on_attack_samples_map: dict[str, float] = {}

    @abstractmethod 
    def _calculate_predictions(self) -> None:
        pass

    def exec(self) -> OneClassResults:
        self._set_vectors_and_true_labels()
        self._calculate_predictions()
        self._calculate_metrics()

        results = OneClassResults( 
                user_model_acc_on_genuine_samples_map = self._user_model_acc_on_genuine_samples_map,
                user_model_recall_map = self._user_model_recall_map,
                user_model_tn_rate_on_attack_samples_map = self._user_model_tn_rate_on_attack_samples_map,
                predictions_on_user_samples_map = self._predictions_on_genuine_samples_map,
                predictions_on_impostor_samples_map = self._predictions_on_attacks_samples_map,
                hp = self._one_class_estimators_hp_map)
        
        return results

    def _set_vectors_and_true_labels(self) -> None:
        for uk in self._dataset.user_keys():
            self._X_training[uk], self._y_training[uk] = self._dataset.one_vs_one_training_rows(uk)
            self._X_test[uk], self._y_test[uk] = self._dataset.one_vs_one_test_rows(uk)
            self._X_attacks[uk], self._y_attacks[uk] = self._dataset.one_vs_one_attacks_rows(uk)
    
    def _calculate_metrics(self):
        for uk in self._dataset.user_keys():
            self._user_model_acc_on_genuine_samples_map[uk] = \
                accuracy_score(self._y_test[uk], self._predictions_on_genuine_samples_map[uk])
            self._user_model_recall_map[uk] = \
                recall_score(self._y_test[uk], self._predictions_on_genuine_samples_map[uk], average='micro')
            self._user_model_tn_rate_on_attack_samples_map[uk] = \
                accuracy_score(self._y_attacks[uk], self._predictions_on_attacks_samples_map[uk])


