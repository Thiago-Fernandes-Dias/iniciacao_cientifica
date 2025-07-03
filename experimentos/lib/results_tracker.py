from datetime import datetime
from lib.one_class_results import ExperimentalResults
from lib.repositories.results_repository import ResultsRepository


class ResultsTracker:
    _results_repo: ResultsRepository
    _exp_start_date: datetime
    _exp_name: str

    def __init__(self, results_repo: ResultsRepository, start_date: datetime, exp_name: str) -> None:
        self._results_repo = ResultsRepository()
        self._exp_start_date = start_date
        self._exp_name = exp_name
    
    def add_result(self, result: ExperimentalResults) -> None:
        result.date = self._exp_start_date
        self._results_repo.add_one_class_result(result, self._exp_name)