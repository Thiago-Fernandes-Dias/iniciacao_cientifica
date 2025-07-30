from abc import abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from lib.one_class_results import ExperimentalResults


class ResultsRepository:
    @abstractmethod
    def add_predictions_frame(self, predictions_frame: pd.DataFrame, seed: int, exp_name: str, date: datetime) -> None:
        pass

    @abstractmethod
    def add_hp(self, hp: dict[str, list[dict[str, Any]]], exp_name: str, date: datetime):
        pass

    @abstractmethod
    def get_one_class_results(self, exp_name: str) -> list[ExperimentalResults]:
        pass
