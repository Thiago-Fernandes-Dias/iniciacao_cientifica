from lib.repositories.results_repository_factory import results_repository_factory

repo = results_repository_factory()
one_class_lw_results = repo.get_one_class_results("one_class_lw_keyrecs")
print(one_class_lw_results[1].user_model_predictions.head())