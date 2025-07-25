from lib.repositories.results_repository_factory import results_repository_factory
from os.path import join, isdir
from os import listdir
import os

repo = results_repository_factory()

directory_path = "./results"

experiments = [f for f in listdir(directory_path) if isdir(join(directory_path, f))]
    
for exp in experiments:
    result = repo.read_results(exp)
    if result is None:
        print(f"No results for {exp}")
    else:
        print(result.model_predictions.show())
        