from typing import Callable

from lib.constants import CMU_PATH, KEYRECS_PATH
from lib.datasets.cmu_dataset import CMUDataset
from lib.datasets.dataset import Dataset
from lib.datasets.keyrecs_dataset import KeyrecsDataset
from lib.repositories.results_repository import ResultsRepository
from lib.runners.experiment_runner import ExperimentRunner
from lib.utils import cmu_test_split, keyrecs_test_split


class ExperimentExecutor:
    _name: str
    _results_repo: ResultsRepository
    _runner_factory: Callable[[Dataset], ExperimentRunner]

    def __init__(self, name: str, results_repo: ResultsRepository,
                 runner_factory: Callable[[Dataset], ExperimentRunner]):
        self._name = name
        self._results_repo = results_repo
        self._runner_factory = runner_factory

    def execute(self) -> None:
        cmu = CMUDataset(CMU_PATH, cmu_test_split)
        cmu_runner = self._runner_factory(cmu)
        results_with_cmu = cmu_runner.exec()
        self._results_repo.add_one_class_result(results_with_cmu, f"{self._name}_cmu")
        keyrecs = KeyrecsDataset(KEYRECS_PATH, keyrecs_test_split)
        keyrecs_runner = self._runner_factory(keyrecs)
        results_with_keyrecs = keyrecs_runner.exec()
        self._results_repo.add_one_class_result(results_with_keyrecs, f"{self._name}_keyrecs")
