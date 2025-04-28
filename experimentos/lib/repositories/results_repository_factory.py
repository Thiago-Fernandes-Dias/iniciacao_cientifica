from lib.repositories.json_results_repository import JsonResultsRepository

def results_repository_factory():
    return JsonResultsRepository()