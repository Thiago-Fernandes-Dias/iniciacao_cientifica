import pickle
import os
import pandas as pd

from lib.one_class_results import ExperimentalResults
from lib.repositories.results_repository import ResultsRepository
from lib.utils import create_dir_if_not_exists

class ParquetResultsRepository(ResultsRepository):
    def add_one_class_result(self, result: ExperimentalResults, exp_name: str, seed: int) -> None:
        path = f"results/{exp_name}/{result.date.strftime('%Y-%m-%d_%H-%M-%S')}"
        create_dir_if_not_exists(path)
        result.model_predictions.to_parquet(f"{path}/predictions.parquet", index=True, engine='fastparquet')
        with open(f"{path}/hp.pickle", "wb") as f:
            pickle.dump(result.hp, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_one_class_results(self, exp_name: str) -> list[ExperimentalResults]:
        results = []
        exp_results_path = f"results/{exp_name}"
        for exp_dir in os.listdir(exp_results_path):
            file_path = f"{exp_results_path}/{exp_dir}"
            if os.path.isdir(file_path):
                try:
                    user_model_predictions = pd.read_parquet(f"{file_path}/predictions.parquet")
                    hp = {}
                    with open(f"{file_path}/hp.pickle", "rb") as f:
                        hp = pickle.load(f)
                    date = pd.to_datetime(exp_dir, format='%Y-%m-%d_%H-%M-%S')
                    results.append(ExperimentalResults(user_model_predictions=user_model_predictions, hp=hp, date=date))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        return sorted(results, key=lambda r: r.date)

