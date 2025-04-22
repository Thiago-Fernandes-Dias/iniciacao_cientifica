from abc import abstractmethod
from lib.one_class_results import OneClassResults

class ResultsRepository:
    @abstractmethod
    def add_one_class_result(self, result: OneClassResults, exp_name: str) -> None:
        pass

    @abstractmethod
    def get_one_class_results(self, exp_name: str) -> list[OneClassResults]:
        pass
