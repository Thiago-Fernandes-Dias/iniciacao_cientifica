from lib.repositories.parquet_results_repository import ParquetResultsRepository

def results_repository_factory():
    return ParquetResultsRepository()