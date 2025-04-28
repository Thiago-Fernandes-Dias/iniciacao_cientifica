import os
from sklearn.svm import OneClassSVM
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.mongo_results_repository import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl

def main() -> None:
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: OneClassExperimentRunnerImpl(
            dataset=ds,
            estimator=OneClassSVM(),
            use_impostor_samples=False
        ),
    )
    executor.execute()

if __name__ == "__main__":
    main()