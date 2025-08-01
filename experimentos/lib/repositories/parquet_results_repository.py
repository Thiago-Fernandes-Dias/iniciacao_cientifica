import logging
from datetime import datetime
import pickle
import os
from logging import log
from typing import Any
import pandas as pd
from icecream import ic
from pyspark.sql import SparkSession
from lib.one_class_results import ExperimentalResults
from lib.repositories.results_repository import ResultsRepository
from lib.utils import create_dir_if_not_exists

class ParquetResultsRepository(ResultsRepository):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add_predictions_frame(self, predictions_frame: pd.DataFrame, seed: int, exp_name: str, date: datetime) -> None:
        results_dir = f"results/{exp_name}/{date.strftime('%Y-%m-%d_%H-%M-%S')}/predictions"
        create_dir_if_not_exists(results_dir)
        predictions_file_path = f"{results_dir}/predictions_{seed}.parquet"
        predictions_frame.to_parquet(predictions_file_path, index=True, engine='fastparquet')
        self.logger.info(f"Model \"{exp_name}\" predictions with seed {seed} saved on {predictions_file_path}.")

    def add_hp(self, hp: dict[str, list[dict[str, Any]]], exp_name: str, date: datetime):
        results_dir = f"results/{exp_name}/{date.strftime('%Y-%m-%d_%H-%M-%S')}"
        create_dir_if_not_exists(results_dir)
        hp_file_path = f"{results_dir}/hp.pickle"
        with open(hp_file_path, "wb") as f:
            pickle.dump(hp, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Model \"{exp_name}\" hyperparameters per seed saved on {hp_file_path}.")

    def read_results(self, exp_name: str) -> ExperimentalResults:
        spark = SparkSession.builder.appName("IC_Keystroke_Dynamics").getOrCreate()
        exp_results_path = f"results/{exp_name}"
        result_dirs = sorted(os.listdir(exp_results_path), key=self._to_datetime)
        if len(result_dirs) == 0:
            return None
        exp_dir = result_dirs[0]
        file_path = f"{exp_results_path}/{exp_dir}"
        if not os.path.isdir(file_path):
            return None
        try:
            directory_path = f"{file_path}/predictions"
            user_model_predictions = spark.read.parquet(directory_path)
            hp = {}
            with open(f"{file_path}/hp.pickle", "rb") as f:
                hp = pickle.load(f)
            date = self._to_datetime(exp_dir)
            return ExperimentalResults(user_model_predictions=user_model_predictions, hp=hp, date=date)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
