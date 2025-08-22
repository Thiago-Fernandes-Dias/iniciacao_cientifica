from typing import Callable

from lib.datasets.cmu_dataset import CMUDataset
from lib.datasets.dataset import Dataset
from lib.datasets.keyrecs_dataset import KeyrecsDataset
from lib.runners.experiment_runner import ExperimentRunner
from lib.utils import cmu_split, keyrecs_split, CMU_PATH, KEYRECS_PATH


class ExperimentExecutor:
    _runner_factory: Callable[[Dataset], ExperimentRunner]

    def __init__(self, runner_factory: Callable[[Dataset], ExperimentRunner]):
        self._runner_factory = runner_factory

    def execute(self) -> None:
        cmu = CMUDataset(CMU_PATH, cmu_split)
        cmu_runner = self._runner_factory(cmu)
        cmu_runner.add_name_suffix("CMU")
        cmu_runner.exec()
        keyrecs = KeyrecsDataset(KEYRECS_PATH, keyrecs_split)
        keyrecs_runner = self._runner_factory(keyrecs)
        keyrecs_runner.add_name_suffix("Keyrecs")
        keyrecs_runner.exec()
