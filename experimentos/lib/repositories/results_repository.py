from abc import abstractmethod
from lib.one_class_results import ExperimentalResults

class ResultsRepository:
    @abstractmethod
    def add_one_class_result(self, result: ExperimentalResults, exp_name: str) -> None:
        pass

    @abstractmethod
    def get_one_class_results(self, exp_name: str) -> list[ExperimentalResults]:
        pass
