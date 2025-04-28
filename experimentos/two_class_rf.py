import os

from sklearn.ensemble import RandomForestClassifier

from lib.datasets.dataset import Dataset
from lib.repositories.results_repository import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import cmu_first_session_split,  save_results


def main() -> None:
    cmu_database = Dataset('datasets/cmu/DSL-StrongPasswordData.csv', cmu_first_session_split)
    exp = OneClassExperimentRunnerImpl(dataset=cmu_database,
                                       estimator=RandomForestClassifier(), use_impostor_samples=True)
    results = exp.exec()
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", ""))
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", ""))

if __name__ == "__main__":
    main()