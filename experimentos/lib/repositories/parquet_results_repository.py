from datetime import datetime
import pickle
import os
from typing import Any
import pandas as pd

from lib.one_class_results import ExperimentalResults
from lib.repositories.results_repository import ResultsRepository
from lib.utils import create_dir_if_not_exists

class ParquetResultsRepository(ResultsRepository):
    def add_predictions_frame(self, predictions_frame: pd.DataFrame, seed: int, exp_name: str, date: datetime) -> None:
        path = f"results/{exp_name}/{date.strftime('%Y-%m-%d_%H-%M-%S')}"
        create_dir_if_not_exists(path)
        predictions_frame.to_parquet(f"{path}/predictions_{seed}.parquet", index=True, engine='fastparquet')
    
    def add_hp(self, hp: dict[str, list[dict[str, Any]]], exp_name: str, date: datetime):
        path = f"results/{exp_name}/{date.strftime('%Y-%m-%d_%H-%M-%S')}"
        create_dir_if_not_exists(path)
        with open(f"{path}/hp.pickle", "wb") as f:
            pickle.dump(hp, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_one_class_results(self, exp_name: str) -> list[ExperimentalResults]:
        pass

