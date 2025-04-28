import os
from sklearn.svm import OneClassSVM
from lib.dataset import Dataset
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import cmu_first_session_split

def main() -> None:
    cmu_database = Dataset('datasets/cmu/DSL-StrongPasswordData.csv', cmu_first_session_split)
    one_class_svm_experiment = OneClassExperimentRunnerImpl(dataset=cmu_database, estimator=OneClassSVM())
    results = one_class_svm_experiment.exec()
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", ""))

if __name__ == "__main__":
    main()