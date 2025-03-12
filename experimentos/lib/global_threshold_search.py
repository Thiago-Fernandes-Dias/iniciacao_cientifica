from sklearn.base import BaseEstimator

from lib.cmu_dataset import CMUDataset
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl


class GlobalThresholdTuning:
    _dataset: CMUDataset
    _estimator: BaseEstimator
    _thresholds: list[float]

    def __init__(self, dataset: CMUDataset, estimator: BaseEstimator, thresholds: list[float]):
        self._dataset = dataset
        self._estimator = estimator
        self._thresholds = thresholds

    def search(self) -> float:
        best_threshold: float = 0
        best_bacc: float = 0
        for threshold in self._thresholds:
            one_class_lw_experiment = OneClassExperimentRunnerImpl(
                dataset=self._dataset,
                estimator=self._estimator.set_params(**{"threshold": threshold})
            )
            results = one_class_lw_experiment.exec()
            g_recall = 1 - results.get_average_frr()
            i_recall = 1 - results.get_average_far()
            bacc = (g_recall + i_recall) / 2
            if bacc > best_bacc:
                best_threshold = threshold
                best_bacc = bacc
        return best_threshold