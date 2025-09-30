from lib.repositories.results_repository_factory import results_repository_factory
from icecream import ic

repo = results_repository_factory()

results = repo.read_results("SVM (Keyrecs)")

ic(results.hp_per_seed[0])