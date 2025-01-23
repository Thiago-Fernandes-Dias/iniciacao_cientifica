from abc import abstractmethod
from sklearn.base import BaseEstimator
from cmu_dataset import CMUDataset
from one_class_results import *

class OneClassExperiment:
    _dataset: CMUDataset
    _estimator: BaseEstimator

    _X_training: dict[str, pd.DataFrame] = {}
    _X_test: dict[str, pd.DataFrame] = {}
    _y_training: dict[str, list[int]] = {}
    _y_test: dict[str, list[int]] = {}
    _X_attacks: dict[str, pd.DataFrame] = {}
    _y_attacks: dict[str, list[int]] = {}

    def __init__(self, dataset: CMUDataset, estimator: BaseEstimator):
        self._dataset = dataset
        self._estimator = estimator

    @abstractmethod
    def exec(self, *, include_hpo: bool = False) -> OneClassResults:
        pass

    def _set_vectors_and_predictions(self) -> None:
        for uk in self._dataset.user_keys():
            self._X_training[uk], self._y_training[uk] = self._dataset.one_vs_one_training_rows(uk)
            self._X_test[uk], self._y_test[uk] = self._dataset.one_vs_one_test_rows(uk)
            self._X_attacks[uk], self._y_attacks[uk] = self._dataset.one_vs_one_attacks_rows(uk)
