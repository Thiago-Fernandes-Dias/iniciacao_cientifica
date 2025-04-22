from pymongo import MongoClient
from lib.constants import MONGO_CONN_STRING
from lib.one_class_results import OneClassResults
from lib.repositories.results_repository import ResultsRepository

class MongoResultsRepository(ResultsRepository):
    def __init__(self, mongo_client):
        self._mongo_client = mongo_client
        self._db = self._mongo_client["exp_results"]

    def add_one_class_result(self, result: OneClassResults, exp_name: str) -> None:
        self._db[exp_name].insert_one(result.to_dict_with_stats())

    def get_one_class_results(self, exp_name: str) -> list[OneClassResults]:
        results = list(self._db[exp_name].find(sort=[("date", -1)])) 
        return list(map(lambda x: OneClassResults.from_dict(x), results))

def results_repository_factory():
    mongo_client = MongoClient(MONGO_CONN_STRING)
    return MongoResultsRepository(mongo_client)