import logging
import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession

from lib.experiment_results import ExperimentResults
from lib.repositories.results_repository import ResultsRepository
from lib.utils import create_dir_if_not_exists


class ParquetResultsRepository(ResultsRepository):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add_predictions_frame(self, *, predictions_frame: pd.DataFrame, seed: int = 0, exp_name: str, date: datetime) -> None:
        results_dir = f"results/{exp_name}/{date.strftime('%Y-%m-%d_%H-%M-%S')}/predictions"
        create_dir_if_not_exists(results_dir)
        predictions_file_path = f"{results_dir}/predictions_{seed}.parquet"
        predictions_frame.to_parquet(predictions_file_path, index=True, engine='fastparquet')
        self.logger.info(f"Experiment \"{exp_name}\" predictions with seed {seed} saved on {predictions_file_path}.")

    def add_hp(self, hp: dict[str, Any], exp_name: str, date: datetime, seed: int = 0) -> None:
        results_dir = f"results/{exp_name}/{date.strftime('%Y-%m-%d_%H-%M-%S')}/hp"
        create_dir_if_not_exists(results_dir)
        hp_file_path = f"{results_dir}/hp_{seed}.pickle"
        with open(hp_file_path, "wb") as f:
            pickle.dump(hp, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"\"{exp_name}\" hyperparameters with seed {seed} saved on {hp_file_path}.")

    def read_results(self, exp_name: str) -> ExperimentResults | None:
        exp_results_path = f"results/{exp_name}"
        result_dirs = sorted(os.listdir(exp_results_path), key=self._to_datetime, reverse=True)
        if len(result_dirs) == 0:
            return None
        current_exp_results_dir = result_dirs[0]
        file_path = f"{exp_results_path}/{current_exp_results_dir}"
        if not os.path.isdir(file_path):
            return None
        try:
            predictions_dir = f"{file_path}/predictions"
            model_predictions_per_seed = []
            for file in os.listdir(predictions_dir):
                if file.endswith(".parquet"):
                    predictions_df = pd.read_parquet(os.path.join(predictions_dir, file), engine='fastparquet')
                    model_predictions_per_seed.append(predictions_df)
            hp_per_seed = []
            # FIXME: comentado para evitar erro de leitura dos resultados atuais.
            # hps_dir = f"{file_path}/hp"
            # for file in os.listdir(hps_dir):
            #     if file.endswith(".pickle"):
            #         with open(os.path.join(hps_dir, file), "rb") as f:
            #             hp = pickle.load(f)
            #             hp_per_seed.append(hp)
            date = self._to_datetime(current_exp_results_dir)
            return ExperimentResults(model_predictions_per_seed=model_predictions_per_seed, hp_per_seed=hp_per_seed,
                                       date=date)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _to_datetime(self, s: str):
        return pd.to_datetime(s, format='%Y-%m-%d_%H-%M-%S')