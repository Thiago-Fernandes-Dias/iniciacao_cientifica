import datetime
import json
import uuid
from lib.one_class_results import OneClassResults
from lib.repositories.results_repository import ResultsRepository
from lib.utils import create_dir_if_not_exists
import os

class JsonResultsRepository(ResultsRepository):
    def add_one_class_result(self, result: OneClassResults, exp_name: str) -> None:
        create_dir_if_not_exists(f"results/{exp_name}")
        json_string = json.dumps(result.to_dict_with_stats(), indent=4, ensure_ascii=True, sort_keys=True)
        with open(f"results/{exp_name}/{uuid.uuid4()}.json", "w+") as f:
            f.write(json_string)
    
    def get_one_class_results(self, exp_name: str):
        results = []
        folder_path = f"results/{exp_name}"
        if not os.path.exists(folder_path):
            return results
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".json"):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    results.append(OneClassResults.from_dict(data))
        return sorted(results, key = lambda r: r.date)