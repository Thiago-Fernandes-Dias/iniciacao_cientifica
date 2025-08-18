from abc import abstractmethod
from datetime import datetime
from typing import Any
import pandas as pd

from lib.experiment_results import ExperimentResults


class ResultsRepository:
    @abstractmethod
    def add_predictions_frame(self, predictions_frame: pd.DataFrame, seed: int, exp_name: str, date: datetime) -> None:
        pass

    @abstractmethod
    def add_hp(self, hp: dict[str, Any], exp_name: str, date: datetime, seed: int = 0) -> None:
        pass

    @abstractmethod
    def read_results(self, exp_name: str) -> ExperimentResults:
        pass

